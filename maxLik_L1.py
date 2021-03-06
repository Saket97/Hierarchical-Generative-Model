import tensorflow as tf
import numpy as np
import pickle as pkl
import math
import os
import sys
import argparse

slim = tf.contrib.slim
ds = tf.contrib.distributions
st = tf.contrib.bayesflow.stochastic_tensor
graph_replace = tf.contrib.graph_editor.graph_replace

""" Hyperparameters """
latent_dim=60
nepoch = 7001
lr = 10**(-4)
batch_size = 160
ntrain = 160
test_batch_size = 52
ntest=52
inp_data_dim = 5000
inp_cov_dim = 7
eps_dim = 40
eps_nbasis=32
n_clf_epoch = 5000
thresh_adv = 0.5
rank = 20

""" tensorboard """
parser = argparse.ArgumentParser()
parser.add_argument('--logdir',type=str,default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),'tensorflow/mnist/logs/mnist_with_summaries'),help='Summaries log directory')
FLAGS, unparsed = parser.parse_known_args()
if tf.gfile.Exists(FLAGS.logdir):
    tf.gfile.DeleteRecursively(FLAGS.logdir)
tf.gfile.MakeDirs(FLAGS.logdir)

def load_dataset():
    raw_data = np.load("/opt/data/saket/gene_data/data/mod_total_data.npy")
    cov = np.load("/opt/data/saket/gene_data/data/cov.npy")
    labels = np.load("/opt/data/saket/gene_data/data/data_label.npy")
    inp_data_dim = raw_data.shape[1]
    inp_cov_dim = cov.shape[1]
    m = np.mean(raw_data, axis=0)
    raw_data = (raw_data-m)/5.0
    cov = (np.log10(cov+0.1))/5.0
    return raw_data[0:ntrain],cov[0:ntrain],labels[0:ntrain],raw_data[ntrain:],cov[ntrain:],labels[ntrain:]

X_train, C_train, L_train, X_test, C_test, L_test = load_dataset()
X_test = X_test.astype(np.float32)
C_test = C_test.astype(np.float32)

l1_regulariser = tf.contrib.layers.l1_regularizer(0.05)

def variable_summaries(var, name=None):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('%s mean'%name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('%s stddev'%name, stddev)
        tf.summary.scalar('%s max'%name, tf.reduce_max(var))
        tf.summary.scalar('%s min'%name, tf.reduce_min(var))
        tf.summary.histogram('%s histogram'%name, var)

def get_zlogprob(z, z_dist):
    if z_dist == "gauss":
        logprob = -0.5 * tf.reduce_sum(z*z  + np.log(2*np.pi), [1])
    elif z_dist == "uniform":
        logprob = 0.
    else:
        raise ValueError("Invalid parameter value for `z_dist`.")
    return logprob

def lrelu(x, leak=0.2, name="lrelu"):
    """ Leaky relu activation function """
    return tf.maximum(x, leak*x)
def sample_gumbel(shape, eps=1e-20):
    """ Sample from Gumbel(0,1)"""
    U = tf.random_uniform(shape, minval=0,maxval=1)
    return -tf.log(-tf.log(U+eps) + eps)

def gumble_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution """
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y/tf.expand_dims(temperature, axis=1))

def gumbel_softmax(logits, temperature, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
            logits: [batch_size, n_class] unnormalized log-probs
            temperature: non-negative scalar
            hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
            [batch_size, n_class] sample from the Gumbel-Softmax distribution.
            If hard=True, then the returned sample will be one-hot, otherwise it will
            be a probabilitiy distribution that sums to 1 across classes
    """

    y = gumble_softmax_sample(logits, temperature)
    return y

def cloglog(x):
    return 1-tf.exp(-tf.exp(10*x))
    #return tf.sigmoid(20*x)

def standard_normal(shape, **kwargs):
    """Create a standard Normal StochasticTensor."""
    return tf.cast(st.StochasticTensor(
        ds.MultivariateNormalDiag(tf.zeros(shape), tf.ones(shape), **kwargs)),  tf.float32)

def data_network(z, n_layer=1, n_hidden=256, reuse=False):
    """ Calculates the value of log(r(z_2|x)/q(z_2|x))"""
    with tf.variable_scope("data_net", reuse = reuse):
        h = slim.repeat(z, n_layer, slim.fully_connected, n_hidden, activation_fn=tf.nn.elu, weights_regularizer=slim.l2_regularizer(0.01))
        out = slim.fully_connected(h, 1, activation_fn=None, weights_regularizer=slim.l2_regularizer(0.01))
    #out = tf.Print(out, [out], message="data_net_out")
    return tf.squeeze(out)

def U_ratio(U, n_layer=2, n_hidden=128, reuse=False):
    """ U:(n_samples, inp_data_dim, rank) """
    with tf.variable_scope("U", reuse=reuse):
        U = tf.reshape(U, [-1, inp_data_dim*rank])
        h = slim.repeat(U,n_layer,slim.fully_connected,n_hidden,activation_fn=lrelu,weights_regularizer=slim.l2_regularizer(0.1))
        h = slim.fully_connected(h,1,activation_fn=None,weights_regularizer=slim.l2_regularizer(0.1))
        variable_summaries(h,name="U_ratio")
    return h

def V_ratio(U, n_layer=2, n_hidden=128, reuse=False):
    """ U:(n_samples, inp_data_dim, rank) """
    with tf.variable_scope("V", reuse=reuse):
        U = tf.reshape(U, [-1, latent_dim*rank])
        h = slim.repeat(U,n_layer,slim.fully_connected,n_hidden,activation_fn=lrelu,weights_regularizer=slim.l2_regularizer(0.1))
        h = slim.fully_connected(h,1,activation_fn=None,weights_regularizer=slim.l2_regularizer(0.1))
        variable_summaries(h,name="V_ratio")
    return h

def B_ratio(U, n_layer=2, n_hidden=128, reuse=False):
    """ U:(n_samples, inp_data_dim, rank) """
    with tf.variable_scope("B", reuse=reuse):
        U = tf.reshape(U, [-1, inp_data_dim*inp_cov_dim])
        h = slim.repeat(U,n_layer,slim.fully_connected,n_hidden,activation_fn=lrelu,weights_regularizer=slim.l2_regularizer(0.1))
        h = slim.fully_connected(h,1,activation_fn=None,weights_regularizer=slim.l2_regularizer(0.1))
        variable_summaries(h,name="B_ratio")
    return h

def D_ratio(D, n_layer=2, n_hidden=128, reuse=False):
    with tf.variable_scope("del", reuse=reuse):
        h = slim.repeat(D,n_layer, slim.fully_connected,n_hidden,activation_fn=lrelu,weights_regularizer=slim.l2_regularizer(0.1))
        h = slim.fully_connected(h,1,activation_fn=None,weights_regularizer=slim.l2_regularizer(0.1))
        variable_summaries(h,name="D_ratio")
    return h

def M_ratio(M, n_layer=2, n_hidden=128, reuse=False):
    with tf.variable_scope("M", reuse=reuse):
        M = tf.reshape(M, [-1, inp_data_dim*latent_dim])
        M = M+tf.random_normal(tf.shape(M))
        variable_summaries(M,name="M_ratio_input")
        #M = tf.Print(M,[M],message="M input")
        h = slim.repeat(M,n_layer, slim.fully_connected,n_hidden,activation_fn=lrelu,weights_regularizer=slim.l2_regularizer(0.1))
        h = slim.fully_connected(h,1,activation_fn=None,weights_regularizer=slim.l2_regularizer(0.1))
        #h = tf.Print(M,[M],message="M ratio network output...")
        variable_summaries(h,name="M_ratio")
   
    return h

def encoder(x,c, eps=None, n_layer=1, n_hidden=128, reuse=False):
    with tf.variable_scope("Encoder", reuse = reuse):
        h = tf.concat([x,c], axis=1)
        h = slim.repeat(h, n_layer, slim.fully_connected, n_hidden, activation_fn=tf.nn.elu, weights_regularizer=slim.l2_regularizer(0.1))
        out = slim.fully_connected(h, latent_dim, activation_fn=None, weights_regularizer=slim.l2_regularizer(0.1))
        #out = tf.Print(out,[out],message="Encoder:out")
        a_vec = []
        for i in range(eps_nbasis):
            a = slim.fully_connected(h, latent_dim, activation_fn=None, scope='a_%d'%i)
            a = tf.nn.elu(a-5.)+1.0
            a_vec.append(a)

        if eps == None:
            eps = tf.random_normal(tf.stack([eps_nbasis, batch_size, eps_dim]))
        
        v_vec = []
        for i in range(eps_nbasis):
            with tf.variable_scope("eps_%d"%i):
                v = slim.repeat(eps[i], 3, slim.fully_connected, 128, activation_fn=tf.nn.elu)
                v = slim.fully_connected(v,latent_dim,activation_fn=None)
                v_vec.append(v)
        
        z = out
        Ez = out
        Varz = 0.0

        for a,v in zip(a_vec, v_vec):
            z += a*v
            Ev, Varv = tf.nn.moments(v, [0])
            Ez += a*Ev
            Varz += a*a*Varv
    #z = tf.Print(z,[z],message="Encoder:z")
    #Ez = tf.Print(Ez,[Ez],message="Ez")
    #Varz = tf.Print(Varz,[Varz],message="Varz")
    return z,Ez,Varz

def generator(n_samples=1, noise_dim=100, reuse=False):
    """ Generate samples for A,B and DELTA 
        Returns:
            A: A tensor of shape (n_samples, inp_data_dim, latent_dim)
            B: A tensor of shape (n_samples, inp_data_dim, inp_cov_dim)
            D: A tensor of shape (n_samples, inp_data_dim)
    """
    with tf.variable_scope("generator",reuse=reuse):
        w = tf.random_normal([n_samples, noise_dim], mean=0, stddev=1.0)
        out = slim.fully_connected(w,512,activation_fn=tf.nn.elu, weights_regularizer=slim.l2_regularizer(0.05)) # (1,1024)

        u = slim.fully_connected(out, 256, activation_fn=tf.nn.elu, weights_regularizer=slim.l2_regularizer(0.05))
        u = slim.fully_connected(u,inp_data_dim*rank,activation_fn=None,weights_regularizer=slim.l2_regularizer(0.05))
        U = tf.reshape(u, [-1,inp_data_dim,rank])
        variable_summaries(U, name="U_generator")
        v = slim.fully_connected(out, 256, activation_fn=tf.nn.elu, weights_regularizer=slim.l2_regularizer(0.05))
        v = slim.fully_connected(v,latent_dim*rank,activation_fn=None,weights_regularizer=slim.l2_regularizer(0.05))        
        V = tf.reshape(v,[-1,rank,latent_dim])
        print("V in generator:",v.get_shape().as_list())
        variable_summaries(V, name="V_generator")

        b = slim.fully_connected(out, 256, activation_fn=tf.nn.elu, weights_regularizer=slim.l2_regularizer(0.1))
        b = slim.fully_connected(b,inp_data_dim*inp_cov_dim,activation_fn=None,weights_regularizer=slim.l2_regularizer(0.1))
        B = tf.reshape(b, [-1,inp_data_dim,inp_cov_dim])

        h = slim.fully_connected(out,1024,activation_fn=tf.nn.elu, weights_regularizer=slim.l2_regularizer(0.1))
        del_sample = slim.fully_connected(h,inp_data_dim,activation_fn=None, weights_regularizer=slim.l2_regularizer(0.1))

        # Sample M
        u = slim.fully_connected(out, 256, activation_fn=lrelu, weights_regularizer=slim.l2_regularizer(0.005), weights_initializer=tf.orthogonal_initializer(gain=1))
        u = slim.fully_connected(u,inp_data_dim*rank,activation_fn=None,weights_regularizer=slim.l2_regularizer(0.005))
        U_gumbel = tf.reshape(u, [-1,inp_data_dim,rank])

        v = slim.fully_connected(out, 256, activation_fn=lrelu, weights_regularizer=slim.l2_regularizer(0.005), weights_initializer=tf.orthogonal_initializer(gain=1))
        v = slim.fully_connected(v,latent_dim*rank,activation_fn=None,weights_regularizer=slim.l2_regularizer(0.005))
        V_gumbel = tf.reshape(v,[-1,rank,latent_dim])

        #t = slim.fully_connected(out, 256, activation_fn=lrelu, weights_regularizer=slim.l2_regularizer(0.001), weights_initializer=tf.orthogonal_initializer(gain=1.0))
        #t = slim.fully_connected(t,1,activation_fn=tf.sigmoid,weights_regularizer=slim.l2_regularizer(0.001), weights_initializer=tf.orthogonal_initializer(gain=1.0))
        #t = tf.Print(t,[t],message="Temperature...")
        #variable_summaries(t, name="Temperature")
        #t = tf.reshape(t,[-1,1,1])
        #t = tf.constant(0.5, dtype=tf.float32,shape=[n_samples,1,1])
        logits_gumbel = tf.transpose(tf.matmul(U_gumbel,V_gumbel),perm=[0,1,2]) #(nsamples,inp_data_dim,latent_dim,2)
        #logits_gumbel = tf.Print(logits_gumbel,[logits_gumbel],message="logits_gumbel")
        variable_summaries(logits_gumbel,name="logits_gumbel")
        #M = gumbel_softmax(logits_gumbel, t)
        #M = tf.squeeze(tf.slice(M, [0,0,0,0],[-1,-1,-1,1]), axis=3)
       # M = tf.transpose(M,perm=[0,2,1])
        #M = tf.Print(M,[M],message="M output generator")
        M = cloglog(logits_gumbel) 
        variable_summaries(M,name="M_generator")
    return U,V,B,del_sample,M

def decoder_mean(z,c, n_hidden=256, n_layers=2, reuse=False):
    """ Convert the linear decoder to non-linerar """
    with tf.variable_scope("decoder",reuse=reuse):
        with tf.variable_scope("z_only",reuse=reuse):
            zh = slim.repeat(z,n_layers,slim.fully_connected,n_hidden,weights_regularizer=slim.l2_regularizer(0.1))
        with tf.variable_scope("c_only",reuse=reuse): 
            h = slim.repeat(c,n_layers,slim.fully_connected,n_hidden//2,weights_regularizer=slim.l2_regularizer(0.1))
        h = tf.concat([zh,h],axis=1)
        out = slim.fully_connected(h,inp_data_dim,activation_fn = None, weights_regularizer=slim.l2_regularizer(0.1))
    return out

def classifier(x_input,labels, reuse=False):
    with tf.variable_scope("Classifier", reuse=reuse):
        logits = slim.fully_connected(x_input, 3, activation_fn=None, weights_regularizer=slim.l2_regularizer(0.1))
        cl_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    return logits, tf.reduce_mean(cl_loss)

def cal_theta_adv_loss(q_samples_A, q_samples_B, q_samples_D, n_samples = 100):
    # n_samples = 100
    p_samples_U = tf.random_normal([n_samples, inp_data_dim, rank])
    p_samples_V = tf.random_normal([n_samples, rank, latent_dim])
    p_samples_B = tf.random_normal([n_samples, inp_data_dim, inp_cov_dim])
    p_samples_D = tf.random_normal([n_samples, inp_data_dim])
    p_samples_M = gumbel_softmax(tf.ones([n_samples, inp_data_dim, latent_dim,2]), tf.constant(0.01, dtype=tf.float32,shape=(n_samples,1,1)))
    p_samples_M = tf.squeeze(tf.slice(p_samples_M,[0,0,0,0],[-1,-1,-1,1]),axis=3)
    q_samples_U, q_samples_V, q_samples_B, q_samples_D, q_samples_M = generator(n_samples=n_samples, reuse=True)
    variable_summaries(p_samples_M, name="p_samples_M")
    variable_summaries(q_samples_M, name="q_samples_M")
    q_ratio_m = 0

    p_ratio = U_ratio(p_samples_U)
    q_ratio = U_ratio(q_samples_U, reuse=True)
    d_loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=p_ratio, labels=tf.ones_like(p_ratio)))
    d_loss_i = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=q_ratio, labels=tf.zeros_like(q_ratio)))
    correct_labels_adv = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.greater(tf.sigmoid(p_ratio),thresh_adv), tf.int32),1),tf.float32)) + tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.less_equal(tf.sigmoid(q_ratio),thresh_adv), tf.int32),1),tf.float32))
    label_acc_adv_u = correct_labels_adv/(2*n_samples)
    label_acc_adv_u = tf.Print(label_acc_adv_u, [label_acc_adv_u], message="label_acc_adv_u")
    dloss_u = d_loss_d+d_loss_i
    q_ratio_m += tf.reduce_mean(q_ratio)

    p_ratio = V_ratio(p_samples_V)
    q_ratio = V_ratio(q_samples_V, reuse=True)
    d_loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=p_ratio, labels=tf.ones_like(p_ratio)))
    d_loss_i = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=q_ratio, labels=tf.zeros_like(q_ratio)))
    correct_labels_adv = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.greater(tf.sigmoid(p_ratio),thresh_adv), tf.int32),1),tf.float32)) + tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.less_equal(tf.sigmoid(q_ratio),thresh_adv), tf.int32),1),tf.float32))
    print("correct_labels_shape:",p_ratio.get_shape().as_list())
    label_acc_adv_v = correct_labels_adv/(2*n_samples)
    dloss_v = d_loss_d+d_loss_i
    label_acc_adv_v = tf.Print(label_acc_adv_v, [label_acc_adv_v], message="label_acc_adv_v")
    q_ratio_m += tf.reduce_mean(q_ratio)

    p_ratio = B_ratio(p_samples_B)
    q_ratio = B_ratio(q_samples_B, reuse=True)
    d_loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=p_ratio, labels=tf.ones_like(p_ratio)))
    d_loss_i = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=q_ratio, labels=tf.zeros_like(q_ratio)))
    correct_labels_adv = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.greater(tf.sigmoid(p_ratio),thresh_adv), tf.int32),1),tf.float32)) + tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.less_equal(tf.sigmoid(q_ratio),thresh_adv), tf.int32),1),tf.float32))
    label_acc_adv_b = correct_labels_adv/(2*n_samples)
    label_acc_adv_b = tf.Print(label_acc_adv_b, [label_acc_adv_b], message="label_acc_adv_b")
    dloss_b = d_loss_d+d_loss_i
    q_ratio_m += tf.reduce_mean(q_ratio)

    p_ratio = D_ratio(p_samples_D)
    q_ratio = D_ratio(q_samples_D, reuse=True)
    d_loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=p_ratio, labels=tf.ones_like(p_ratio)))
    d_loss_i = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=q_ratio, labels=tf.zeros_like(q_ratio)))
    correct_labels_adv = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.greater(tf.sigmoid(p_ratio),thresh_adv), tf.int32),1),tf.float32)) + tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.less_equal(tf.sigmoid(q_ratio),thresh_adv), tf.int32),1),tf.float32))
    label_acc_adv_d = correct_labels_adv/(2*n_samples)
    label_acc_adv_d = tf.Print(label_acc_adv_d, [label_acc_adv_d], message="label_acc_adv_d")
    dloss_d = d_loss_d+d_loss_i
    q_ratio_m += tf.reduce_mean(q_ratio)

    p_ratio = M_ratio(p_samples_M)
    q_ratio = M_ratio(q_samples_M, reuse=True)
    d_loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=p_ratio, labels=tf.ones_like(p_ratio)))
    d_loss_i = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=q_ratio, labels=tf.zeros_like(q_ratio)))
    correct_labels_adv = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.greater(tf.sigmoid(p_ratio),thresh_adv), tf.int32),1),tf.float32)) + tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.less_equal(tf.sigmoid(q_ratio),thresh_adv), tf.int32),1),tf.float32))
    label_acc_adv_m = correct_labels_adv/(2*n_samples)
    label_acc_adv_m = tf.Print(label_acc_adv_m, [label_acc_adv_m], message="label_acc_adv_m")
    dloss_m = d_loss_d+d_loss_i
    q_ratio_m += tf.reduce_mean(q_ratio)

    dloss = dloss_u + dloss_v + dloss_b + dloss_d + dloss_m
    
    #Adversary Accuracy
    # correct_labels_adv = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.greater(tf.sigmoid(p_ratio),thresh_adv), tf.int32),1),tf.float32)) + tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.less_equal(tf.sigmoid(q_ratio),thresh_adv), tf.int32),0),tf.float32))
    label_acc_adv_theta = (label_acc_adv_b+label_acc_adv_d+label_acc_adv_u+label_acc_adv_v+label_acc_adv_m)/5.0
    label_acc_adv_theta = tf.Print(label_acc_adv_theta, [label_acc_adv_theta], message="label_acc_adv_theta")   
    return dloss, label_acc_adv_theta, q_ratio_m

def train(z, closs, label_acc_adv_theta):
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith("data_net")]
    c_vars = [var for var in t_vars if var.name.startswith("Classifier")]
    tr_vars = [var for var in t_vars if (var.name.startswith("U") or var.name.startswith("V") or var.name.startswith("B") or var.name.startswith("del") or var.name.startswith("M"))]
    tp_var = [var for var in t_vars if var not in d_vars+c_vars+tr_vars]
    print("tp_var:",tp_var)
    print("tr_var:",tr_vars)
    assert(len(tp_var)+len(d_vars)+len(c_vars)+len(tr_vars) == len(t_vars))
    
    r_loss = tf.losses.get_regularization_loss()
    r_loss_clf = tf.losses.get_regularization_loss(scope="Classifier")
    r_loss -= r_loss_clf
    primal_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, use_locking=True, beta1=0.5)
    dual_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, use_locking=True, beta1=0.5)
    dual_optimizer_theta = tf.train.AdamOptimizer(learning_rate=1e-4, use_locking=True, beta1=0.5)
    #classifier_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3, use_locking=True)

    primal_train_op = primal_optimizer.minimize(primal_loss+r_loss+closs+r_loss_clf, var_list=tp_var+c_vars)
    adversary_train_op = dual_optimizer.minimize(dual_loss+r_loss, var_list=d_vars)
    adversary_theta_train_op = dual_optimizer_theta.minimize(dual_loss_theta+r_loss, var_list=tr_vars)
    #clf_train_op = classifier_optimizer.minimize(closs+r_loss_clf, var_list=c_vars)
    train_op = tf.group(primal_train_op, adversary_train_op)
    
    # Test Set Graph
    eps = tf.random_normal(tf.stack([eps_nbasis, test_batch_size,eps_dim]))
    z_test, _, _ = encoder(X_test,C_test,eps,reuse=True)
    U_test,V_test,B_test,D_test,M_test = generator(n_samples=500, reuse=True)
    A = tf.matmul(U_test,V_test)
    A = A*M_test
    means = tf.matmul(tf.ones([A.get_shape().as_list()[0],z_test.get_shape().as_list()[0],latent_dim])*z_test,tf.transpose(A, perm=[0,2,1]))+tf.matmul(tf.ones([B_test.get_shape().as_list()[0],C_test.shape[0],inp_cov_dim])*C_test,tf.transpose(B_test, perm=[0,2,1])) # (n_samples, 52, 60) (n_samples, 60, 5000) = (n_samples, 52, 5000)
    prec = tf.square(D_test)
    t = (X_test-means)
    t1 = t*tf.expand_dims(prec, axis=1)*t
    t1 = -0.5*tf.reduce_sum(t1, axis=2)
    t2 = 0.5*tf.expand_dims(tf.reduce_sum(tf.log(1e-3+prec), axis=1), axis=1)
    t3 = -inp_data_dim*0.5*tf.log(2*math.pi)
    x_post_prob_log_test = t1+t2+t3
    #x_post_prob_log_test = tf.reduce_mean(x_post_prob_log_test, axis=1)
    x_post_prob_log_test = tf.reduce_mean(x_post_prob_log_test, axis=0) # expect wrt theta
    logits_test, closs_test = classifier(z_test,labels,reuse=True)
    prob_test = tf.nn.softmax(logits_test)
    correct_label_pred_test = tf.equal(tf.argmax(logits_test,1),labels)
    label_acc_test = tf.reduce_mean(tf.cast(correct_label_pred_test, tf.float32))

    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.logdir, graph = tf.get_default_graph())
    sess = tf.Session()
    #saver.restore(sess,"/opt/data/saket/gene_data/model_spike/model.ckpt-249")
    sess.run(tf.global_variables_initializer())
    si_list = []
    vtp_list = []
    clf_loss_list = []
    post_test_list = []
    dt_list = []
    test_lik_list1 = []
    test_acc_list = []
    for i in range(nepoch):
        for j in range(ntrain//batch_size):
            xmb = X_train[j*batch_size:(j+1)*batch_size]
            cmb = C_train[j*batch_size:(j+1)*batch_size]
            # sess.run(sample_from_r, feed_dict={x:xmb, c:cmb})
            for gen in range(1):
                sess.run(train_op, feed_dict={x:xmb,c:cmb,labels:L_train[j*batch_size:(j+1)*batch_size]})
            for gen in range(1):
                sess.run(adversary_theta_train_op, feed_dict={x:xmb,c:cmb,labels:L_train[j*batch_size:(j+1)*batch_size]})
            si_loss,vtp_loss,closs_,label_acc_,dloss_theta = sess.run([dual_loss, primal_loss, closs, label_acc, dual_loss_theta], feed_dict={x:xmb,c:cmb,labels:L_train[j*batch_size:(j+1)*batch_size]})
            si_list.append(si_loss)
            clf_loss_list.append((closs_, label_acc_))
            vtp_list.append(vtp_loss)
            dt_list.append(dloss_theta)
        if i%100 == 0:
            Td_,KL_neg_r_q_,x_post_prob_log_,logz_,logr_,d_loss_d_,d_loss_i_,label_acc_adv_,label_acc_adv_theta_,dual_loss_theta_ = sess.run([Td,KL_neg_r_q,x_post_prob_log,logz,logr,d_loss_d,d_loss_i,label_acc_adv, label_acc_adv_theta, dual_loss_theta], feed_dict={x:xmb,c:cmb})
            print("epoch:",i," si:",si_list[-1]," vtp:",vtp_list[-1]," -KL_r_q:",KL_neg_r_q_, " x_post:",x_post_prob_log_," logz:",logz_," logr:",logr_, \
            " d_loss_d:",d_loss_d_, " d_loss_i:",d_loss_i_, " adv_accuracy:",label_acc_adv_, " closs:",closs_," label_acc:",label_acc_, " theta_dual_loss:",dual_loss_theta_, "label_acc_theta:",label_acc_adv_theta_)
        if i%500 == 0:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary = sess.run(merged,feed_dict={x:xmb,c:cmb})
            train_writer.add_run_metadata(run_metadata, 'step %i'%(i))
            train_writer.add_summary(summary, i)

        if i%1000 == 0:
            test_lik_list = []
            test_prob = []
            for i in range(250):
                test_lik = sess.run(x_post_prob_log_test)
                test_lik_list.append(test_lik)
                lt, la = sess.run([prob_test, label_acc_test], feed_dict={labels:L_test})
                test_prob.append(lt)
            avg_test_prob = np.mean(test_prob,axis=0)
            avg_test_acc1 = np.mean((np.argmax(avg_test_prob,axis=1)==L_test))
            print("Average Test Set Accuracy:",avg_test_acc1)
            test_acc_list.append(avg_test_acc1)
            test_lik = np.stack(test_lik_list, axis=1)
            test_lik = np.mean(test_lik, axis=1)
            test_lik = np.mean(test_lik)
            test_lik_list1.append(test_lik)
            x_post_test = sess.run(x_post_prob_log_test)
            post_test_list.append(x_post_test)
            print("test set p(x|z):",x_post_test, "Td:",Td_)
            path = saver.save(sess,FLAGS.logdir+"/model.ckpt",i)
            print("Model saved at ",path)
    # Training complete
    # z_list=[]
    # for i in range(n_clf_epoch):
    #     tmp = []
    #     for j in range(ntrain//batch_size):
    #         xmb = X_train[j*batch_size:(j+1)*batch_size]
    #         cmb = C_train[j*batch_size:(j+1)*batch_size]
    #         sess.run(clf_train_op, feed_dict={x:xmb,c:cmb,labels:L_train[j*batch_size:(j+1)*batch_size]})           
    #         z_ = sess.run(z, feed_dict={x:xmb,c:cmb})
    #         cl_loss_, label_acc_ = sess.run([closs, label_acc], feed_dict={x:xmb,c:cmb,labels:L_train[j*batch_size:(j+1)*batch_size]})
    #         tmp.append(z_)
    #     if i%100 == 0:
    #         print("epoch:",i," closs:",cl_loss_," train_acc:",label_acc_)
    #     train_z = np.stack(tmp, axis=0)
    #     if i < 250:
    #         z_list.append(train_z)
    
    A_,B_,DELTA_inv_,M_test_ = sess.run([A,B,DELTA_inv, M_test])
    M_test_ = M_test_[0]
    np.save("si_loss1.npy", si_list)
    np.save("vtp_loss1.npy", vtp_list)
    np.save("x_post_list1.npy",post_test_list)
    np.save("A1.npy",np.mean(A_, axis=0))
    np.save("B1.npy",np.mean(B_,axis=0))
    np.save("delta_inv1.npy",np.mean(DELTA_inv_,axis=0))
    np.save("clf_loss_list1.npy",clf_loss_list)
    np.save("dloss_theta1.npy",dt_list)
    np.save("test_lik.npy",test_lik_list1)
    np.save("test_acc.npy",test_acc_list)
    np.save("M1.npy",M_test_)
    # Test Set
    z_list = []       
    label_acc_ = sess.run(label_acc_test, feed_dict={labels:L_test})
    print("Test Set label Accuracy:", label_acc_)
    test_prob = []
    test_acc = []

    for  i in range(250):
        lt, la = sess.run([prob_test, label_acc_test], feed_dict={labels:L_test})
        test_prob.append(lt)
        test_acc.append(la)
    avg_test_prob = np.mean(test_prob,axis=0)
    print("avg_test_prob_shape:",avg_test_prob.shape)
    avg_test_acc1 = np.mean((np.argmax(avg_test_prob,axis=1)==L_test))
    avg_test_acc = np.mean(test_acc)
    np.save("test_acc.npy",test_acc)
    np.save("test_prob.npy",test_prob)
    print("Average Test Set Accuracy:",avg_test_acc, " :",avg_test_acc1)
    print("Average test likelihood:", test_lik)

    for i in range(20):
        z_ = sess.run(z_test)
        #print("z_:",z_)
        z_list.append(z_)
    np.save("z_test_list.npy",z_list)

if __name__ == "__main__":
    x = tf.placeholder(tf.float32, shape=(batch_size, inp_data_dim))
    c = tf.placeholder(tf.float32, shape=(batch_size, inp_cov_dim))
    z_sampled = tf.random_normal([batch_size, latent_dim])
    labels = tf.placeholder(tf.int64, shape=(None))
    eps = tf.random_normal([batch_size, eps_dim])

    n_samples=4
    U,V,B,DELTA_inv,M = generator(n_samples=n_samples)
    U1 = tf.slice(U,[0,0,0],[1,-1,-1])
    V1 = tf.slice(V,[0,0,0],[1,-1,-1])
    M1 = tf.slice(M,[0,0,0],[1,-1,-1])
    A1 = tf.matmul(U1,V1)
    A1 = A1*M1
    B1 = tf.slice(B,[0,0,0],[1,-1,-1])
    DELTA_inv1 = tf.slice(DELTA_inv, [0,0],[1,-1])
    U_mean, U_var = tf.nn.moments(U, axes=0)
    U_std = tf.sqrt(U_var)
    V_mean, V_var = tf.nn.moments(V, axes=0)
    V_std = tf.sqrt(V_var)
    B_mean, B_var = tf.nn.moments(B, axes=0)
    B_std = tf.sqrt(B_var)
    D_mean, D_var = tf.nn.moments(DELTA_inv, axes = 0)
    D_std = tf.sqrt(D_var)

    #normalising
    U_norm = (U-U_mean)/(1.0*U_std)
    V_norm = (V-V_mean)/(1.0*V_std)
    B_norm = (B-B_mean)/(1.0*B_std)
    D_norm = (DELTA_inv-D_mean)/(1.0*D_std)

    # Draw samples from posterior q(z2|x)
    print("Sampling from posterior...")
    eps = tf.random_normal(tf.stack([eps_nbasis, batch_size, eps_dim]))
    z, z_mean, z_var = encoder(x,c,eps,reuse=False)
    z_mean, z_var = tf.stop_gradient(z_mean), tf.stop_gradient(z_var)
    z_std = tf.sqrt(z_var + 1e-4)
    # z_sampled = R_std*R_sampled+R_mean
    z_norm = (z-1.0*z_mean)/z_std
    logz = get_zlogprob(z, "gauss")
    logr = -0.5 * tf.reduce_sum(z_norm*z_norm + tf.log(z_var) + latent_dim*np.log(2*np.pi), [1])
    logz = tf.reduce_mean(logz)
    logr = tf.reduce_mean(logr)

    # # reference and prior distribution for theta (A,B,delta)
    # logra = -0.5*tf.reduce_sum(A_norm*A_norm + tf.log(A_var) + np.log(2*np.pi), axis=[1,2])
    # logra = tf.reduce_mean(logra, axis=0)
    # logrb = -0.5*tf.reduce_sum(B_norm*B_norm + tf.log(B_var) + np.log(2*np.pi), axis=[1,2])
    # logrb = tf.reduce_mean(logrb, axis=0)
    # logrd = -0.5*tf.reduce_sum(D_norm*D_norm + tf.log(D_var) + np.log(2*np.pi), axis=[1])
    # logrd = tf.reduce_mean(logrd, axis=0)
    logu = -0.5*tf.reduce_sum(U*U + np.log(2*np.pi), axis=[1,2])
    logu = tf.reduce_mean(logu)
    logv = -0.5*tf.reduce_sum(V*V + np.log(2*np.pi), axis=[1,2])
    logv = tf.reduce_mean(logv)    
    logb = -0.5*tf.reduce_sum(B*B + np.log(2*np.pi), [1,2])
    logb = tf.reduce_mean(logb)
    logd = -0.5*tf.reduce_sum(DELTA_inv*DELTA_inv + np.log(2*np.pi), [1])
    logd = tf.reduce_mean(logd)

    # Evaluating p(x|z)
    means = tf.matmul(tf.ones([A1.get_shape().as_list()[0],z.get_shape().as_list()[0],latent_dim])*z,tf.transpose(A1, perm=[0,2,1]))+tf.matmul(tf.ones([B1.get_shape().as_list()[0],c.get_shape().as_list()[0],inp_cov_dim])*c,tf.transpose(B1, perm=[0,2,1])) # (N,100) (n_samples,5000,100)
    prec = tf.square(DELTA_inv1)
    t = (x-means)
    t1 = t*tf.expand_dims(prec, axis=1)*t
    t1 = -0.5*tf.reduce_sum(t1, axis=2) # (n_samples, batch_size)
    t2 = 0.5*tf.expand_dims(tf.reduce_sum(tf.log(1e-3+prec), axis=1),axis=1) # (n_samples,1)
    t3 = -inp_data_dim*0.5*tf.log(2*math.pi)
    x_post_prob_log = t1+t2+t3
    x_post_prob_log = tf.reduce_mean(x_post_prob_log, axis=1)
    x_post_prob_log = tf.reduce_mean(x_post_prob_log)

    # Classifier
    logits, closs = classifier(z,labels,reuse=False)
    correct_label_pred = tf.equal(tf.argmax(logits,1),labels)
    label_acc = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))

    # Dual loss
    Td = data_network(z_norm)
    Ti = data_network(z_sampled, reuse=True)
    d_loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Td, labels=tf.ones_like(Td)))
    d_loss_i = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Ti, labels=tf.zeros_like(Ti)))
    dual_loss = d_loss_d+d_loss_i

    # dual loss for theta
    dual_loss_theta, label_acc_adv_theta, q_ratio = cal_theta_adv_loss(A1,B,DELTA_inv)
    
    #Adversary Accuracy
    correct_labels_adv = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.greater(tf.sigmoid(Td),thresh_adv), tf.int32),1),tf.float32)) + tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.less_equal(tf.sigmoid(Td),thresh_adv), tf.int32),0),tf.float32))
    label_acc_adv = correct_labels_adv/(2.0*batch_size)

    # Primal loss
    t1 = -tf.reduce_mean(Td)
    t2 = x_post_prob_log+logz-logr
    t3 = q_ratio
    KL_neg_r_q = t1
    ELBO = t1+t2+t3
    primal_loss = tf.reduce_mean(-ELBO)

    print("V in main:",V.get_shape().as_list())
    print("U in main:",U.get_shape().as_list())
    print("B in main:",B.get_shape().as_list())
    print("DELTA_inv in main:",DELTA_inv.get_shape().as_list())
    train(z, closs, label_acc_adv_theta)
