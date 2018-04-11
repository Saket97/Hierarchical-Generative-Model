import tensorflow as tf
import numpy as np
import pickle as pkl
import math
import os
import sys
import argparse
from utils import *

slim = tf.contrib.slim
ds = tf.contrib.distributions
st = tf.contrib.bayesflow.stochastic_tensor
graph_replace = tf.contrib.graph_editor.graph_replace

""" Hyperparameters """
latent_dim=60
nepoch = 13001
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
    raw_data = np.load("/opt/data/saket/gene_data/data/mod_total_data_count.npy")
    cov = np.load("/opt/data/saket/gene_data/data/cov.npy")
    labels = np.load("/opt/data/saket/gene_data/data/data_label.npy")
    global inp_cov_dim
    global inp_data_dim
    inp_data_dim = raw_data.shape[1]
    inp_cov_dim = cov.shape[1]
    XCL = np.concatenate([raw_data,cov,labels], axis=1)
    np.random.shuffle(x)
    raw_data = XCL[:,0:inp_data_dim]
    cov = XCL[:,inp_data_dim:inp_data_dim+inp_cov_dim]
    labels = XCL[:,inp_cov_dim+inp_data_dim:]
    # m = np.mean(raw_data, axis=0)
    # raw_data = (raw_data-m)/5.0
    # cov = (np.log10(cov+0.1))/5.0
    return raw_data[0:ntrain],cov[0:ntrain],labels[0:ntrain],raw_data[ntrain:],cov[ntrain:],labels[ntrain:]

X_train, C_train, L_train, X_test, C_test, L_test = load_dataset()
X_test = X_test.astype(np.float32)
C_test = C_test.astype(np.float32)

l1_regulariser = tf.contrib.layers.l1_regularizer(0.05)


def standard_normal(shape, **kwargs):
    """Create a standard Normal StochasticTensor."""
    return tf.cast(st.StochasticTensor(
        ds.MultivariateNormalDiag(tf.zeros(shape), tf.ones(shape), **kwargs)),  tf.float32)


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
       # M = M+tf.random_normal(tf.shape(M))
        variable_summaries(M,name="M_ratio_input")
        #M = tf.Print(M,[M],message="M input")
        h = slim.repeat(M,n_layer, slim.fully_connected,n_hidden,activation_fn=lrelu,weights_regularizer=slim.l2_regularizer(0.1))
        h = slim.fully_connected(h,1,activation_fn=None,weights_regularizer=slim.l2_regularizer(0.1))
        #h = tf.Print(M,[M],message="M ratio network output...")
        variable_summaries(h,name="M_ratio")
   
    return h

def encoder(x,c, eps=None, n_layer=1, n_hidden=128, reuse=False):
    with tf.variable_scope("Encoder", reuse = reuse):
        x_min = tf.reduce_min(x)
        x_max = tf.reduce_max(x)
        x = -1 + 2*(x-x_min)/(x_max-x_min)
        #x = tf.Print(x,[x],message="x after batch normalising")
        h = tf.concat([x,c], axis=1)
        h = slim.repeat(h, n_layer, slim.fully_connected, n_hidden, activation_fn=tf.nn.elu, weights_regularizer=slim.l2_regularizer(0.5))
        # h = tf.contrib.layers.batch_norm(h,is_training=b_train)
        out = slim.fully_connected(h, latent_dim, activation_fn=None, weights_regularizer=slim.l2_regularizer(0.5))
        # out = tf.contrib.layers.batch_norm(out,is_training=b_train)
        #out = tf.Print(out,[out],message="Encoder:out")
        out = tf.exp(out)
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
                v = tf.exp(v)
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
    print("z shape:",z.get_shape().as_list())
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
        B = tf.exp(B)

        h = slim.fully_connected(out,1024,activation_fn=tf.nn.elu, weights_regularizer=slim.l2_regularizer(0.1))
        del_sample = slim.fully_connected(h,inp_data_dim,activation_fn=None, weights_regularizer=slim.l2_regularizer(0.1))

        # Sample M
        u = slim.fully_connected(out, 256, activation_fn=lrelu, weights_regularizer=slim.l2_regularizer(0.05), weights_initializer=tf.orthogonal_initializer(gain=1.0))
        u = slim.fully_connected(u,inp_data_dim*rank*2,activation_fn=None,weights_regularizer=slim.l2_regularizer(0.05))
        U_gumbel = tf.reshape(u, [-1,2,inp_data_dim,rank])

        v = slim.fully_connected(out, 256, activation_fn=lrelu, weights_regularizer=slim.l2_regularizer(0.05), weights_initializer=tf.orthogonal_initializer(gain=1.0))
        v = slim.fully_connected(v,latent_dim*rank*2,activation_fn=None,weights_regularizer=slim.l2_regularizer(0.05))        
        V_gumbel = tf.reshape(v,[-1,2,rank,latent_dim])

        #t = slim.fully_connected(out, 256, activation_fn=lrelu, weights_regularizer=slim.l2_regularizer(0.001), weights_initializer=tf.orthogonal_initializer(gain=1.0))
        #t = slim.fully_connected(t,1,activation_fn=tf.sigmoid,weights_regularizer=slim.l2_regularizer(0.001), weights_initializer=tf.orthogonal_initializer(gain=1.0))
        #t = tf.Print(t,[t],message="Temperature...")
        #variable_summaries(t, name="Temperature")
        #t = tf.reshape(t,[-1,1,1])
        t = tf.constant(0.5, dtype=tf.float32,shape=[n_samples,1,1])
        logits_gumbel = tf.transpose(tf.matmul(U_gumbel,V_gumbel),perm=[0,2,3,1]) #(nsamples,inp_data_dim,latent_dim,2)
        #logits_gumbel = tf.Print(logits_gumbel,[logits_gumbel],message="logits_gumbel")
        variable_summaries(logits_gumbel,name="logits_gumbel")
        M = gumbel_softmax(logits_gumbel, t)
        M = tf.squeeze(tf.slice(M, [0,0,0,0],[-1,-1,-1,1]), axis=3)
       # M = tf.transpose(M,perm=[0,2,1])
        #M = tf.Print(M,[M],message="M output generator")
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

def C_embed(x, reuse=False):
    with tf.variable_scope("C_embed", reuse=reuse):
        h = slim.fully_connected(x, inp_cov_dim, activation_fn=tf.sigmoid, weights_regularizer=slim.l2_regularizer(0.1))
    return h

def cal_theta_adv_loss(q_samples_A, q_samples_B, q_samples_D, n_samples = 100):
    # n_samples = 100
    p_samples_U = tf.random_normal([n_samples, inp_data_dim, rank])
    p_samples_V = tf.random_normal([n_samples, rank, latent_dim])
    p_samples_B = tf.random_gamma([n_samples, inp_data_dim, inp_cov_dim],1)
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

    primal_grad = primal_optimizer.compute_gradients(primal_loss+5*closs+r_loss+r_loss_clf,var_list=tp_var+c_vars)
    capped_g_grad = []
    for grad, var in primal_grad:
        if grad is not None:
            capped_g_grad.append((tf.clip_by_value(grad, -0.1, 0.1), var))
    primal_train_op = primal_optimizer.apply_gradients(capped_g_grad)
    #primal_train_op = primal_optimizer.minimize(primal_loss+r_loss+closs+r_loss_clf, var_list=tp_var+c_vars)
    adv_grad = dual_optimizer.compute_gradients(dual_loss+r_loss,var_list=d_vars)
    capped_g_grad = []
    for grad, var in adv_grad:
        if grad is not None:
            capped_g_grad.append((tf.clip_by_value(grad, -0.1, 0.1), var))
    adversary_train_op = dual_optimizer.apply_gradients(capped_g_grad)
    #adversary_train_op = dual_optimizer.minimize(dual_loss+r_loss, var_list=d_vars)
    adv_grad = dual_optimizer_theta.compute_gradients(dual_loss_theta+r_loss,var_list=tr_vars)
    capped_g_grad = []
    for grad, var in adv_grad:
        if grad is not None:
            capped_g_grad.append((tf.clip_by_value(grad, -0.1, 0.1), var))
    adversary_theta_train_op = dual_optimizer_theta.apply_gradients(capped_g_grad)
    #adversary_theta_train_op = dual_optimizer_theta.minimize(dual_loss_theta+r_loss, var_list=tr_vars)
    #clf_train_op = classifier_optimizer.minimize(closs+r_loss_clf, var_list=c_vars)
    train_op = tf.group(primal_train_op, adversary_train_op)
    
    # Test Set Graph
    eps = tf.random_normal(tf.stack([eps_nbasis, test_batch_size,eps_dim]))
    z_test, _, _ = encoder(X_test,C_test,eps,reuse=True)
    z_test = tf.Print(z_test,[z_test],message="z_test")
    U_test,V_test,B_test,D_test,M_test = generator(n_samples=500, reuse=True)
    A = tf.matmul(U_test,V_test)
    A = tf.exp(A)
    A = A*M_test
    X_min = tf.reduce_min(X_test)
    X_max = tf.reduce_max(X_test)
    z_test = (z_test+1)*(X_max-X_min)/2.0+X_min
    A = tf.Print(A,[A],message="A_test")
    means = tf.matmul(tf.ones([A.get_shape().as_list()[0],z_test.get_shape().as_list()[0],latent_dim])*z_test,tf.transpose(A, perm=[0,2,1]))+tf.matmul(tf.ones([B_test.get_shape().as_list()[0],C_test.shape[0],inp_cov_dim])*C_embed(C_test,reuse=True),tf.transpose(B_test, perm=[0,2,1])) # (n_samples, 52, 60) (n_samples, 60, 5000) = (n_samples, 52, 5000)
    prec = tf.square(D_test)
    means = means+tf.expand_dims(prec,axis=1)
    means = tf.Print(means,[means],message="means test")
    x_post_prob_log_test = X_test*tf.log(means+1e-5)-means-tf.lgamma(X_test+1)
    x_post_prob_log_test = tf.reduce_mean(x_post_prob_log_test, axis=2, keep_dims=False) # wrt to theta
    x_post_prob_log_test = tf.reduce_mean(x_post_prob_log_test, axis=0, keep_dims=False) # wrt to theta

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
                sess.run(train_op, feed_dict={x:xmb,c:cmb,labels:L_train[j*batch_size:(j+1)*batch_size], b_train:True})
            for gen in range(1):
                sess.run(adversary_theta_train_op, feed_dict={x:xmb,c:cmb,labels:L_train[j*batch_size:(j+1)*batch_size], b_train:True})
            si_loss,vtp_loss,closs_,label_acc_,dloss_theta = sess.run([dual_loss, primal_loss, closs, label_acc, dual_loss_theta], feed_dict={x:xmb,c:cmb,labels:L_train[j*batch_size:(j+1)*batch_size], b_train:True})
            si_list.append(si_loss)
            clf_loss_list.append((closs_, label_acc_))
            vtp_list.append(vtp_loss)
            dt_list.append(dloss_theta)
        if i%100 == 0:
            Td_,KL_neg_r_q_,x_post_prob_log_,logz_,logr_,d_loss_d_,d_loss_i_,label_acc_adv_,label_acc_adv_theta_,dual_loss_theta_ = sess.run([Td,KL_neg_r_q,x_post_prob_log,logz,logr,d_loss_d,d_loss_i,label_acc_adv, label_acc_adv_theta, dual_loss_theta], feed_dict={x:xmb,c:cmb, b_train:True})
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
                test_lik = sess.run(x_post_prob_log_test, feed_dict={b_train:False})
                test_lik_list.append(test_lik)
                lt, la = sess.run([prob_test, label_acc_test], feed_dict={labels:L_test, b_train:False})
                test_prob.append(lt)
            avg_test_prob = np.mean(test_prob,axis=0)
            avg_test_acc1 = np.mean((np.argmax(avg_test_prob,axis=1)==L_test))
            print("Average Test Set Accuracy:",avg_test_acc1)
            test_acc_list.append(avg_test_acc1)
            test_lik = np.stack(test_lik_list, axis=1)
            test_lik = np.mean(test_lik, axis=1)
            test_lik = np.mean(test_lik)
            test_lik_list1.append(test_lik)
            x_post_test = sess.run(x_post_prob_log_test, feed_dict={b_train:False})
            post_test_list.append(x_post_test)
            print("test set p(x|z):",x_post_test, "Td:",Td_)
            path = saver.save(sess,FLAGS.logdir+"/model.ckpt",i)
            print("Model saved at ",path)
    
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
    label_acc_ = sess.run(label_acc_test, feed_dict={labels:L_test, b_train:False})
    print("Test Set label Accuracy:", label_acc_)
    test_prob = []
    test_acc = []

    for  i in range(250):
        lt, la = sess.run([prob_test, label_acc_test], feed_dict={labels:L_test, b_train:False})
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
        z_ = sess.run(z_test, feed_dict={b_train:False})
        #print("z_:",z_)
        z_list.append(z_)
    np.save("z_test_list.npy",z_list)

if __name__ == "__main__":
    x = tf.placeholder(tf.float32, shape=(batch_size, inp_data_dim))
    c = tf.placeholder(tf.float32, shape=(batch_size, inp_cov_dim))
    b_train = tf.placeholder(tf.bool, shape=())
    z_sampled = tf.random_normal([batch_size, latent_dim])
    labels = tf.placeholder(tf.int64, shape=(None))
    eps = tf.random_normal([batch_size, eps_dim])

    n_samples=4
    U,V,B,DELTA_inv,M = generator(n_samples=n_samples)
    U1 = tf.slice(U,[0,0,0],[1,-1,-1])
    V1 = tf.slice(V,[0,0,0],[1,-1,-1])
    M1 = tf.slice(M,[0,0,0],[1,-1,-1])
    A1 = tf.exp(tf.matmul(U1,V1))
    A1 = A1*M1
    B1 = tf.slice(B,[0,0,0],[1,-1,-1])
    DELTA_inv1 = tf.slice(DELTA_inv, [0,0],[1,-1])

    # Draw samples from posterior q(z2|x)
    print("Sampling from posterior...")
    eps = tf.random_normal(tf.stack([eps_nbasis, batch_size, eps_dim]))
    z, z_mean, z_var = encoder(x,c,eps,reuse=False)
    z_mean, z_var = tf.stop_gradient(z_mean), tf.stop_gradient(z_var)
    z_std = tf.sqrt(z_var + 1e-4)
    # z_sampled = R_std*R_sampled+R_mean
    z_norm = (z-1.0*z_mean)/z_std
    logz = get_zlogprob(z, "gauss")
    logr = -0.5 * tf.reduce_sum(z_norm*z_norm + tf.log(z_var+1e-5) + latent_dim*np.log(2*np.pi), [1])
    logz = tf.reduce_mean(logz)
    logr = tf.reduce_mean(logr)

    #z = tf.Print(z,[z],message="z")
    #A1 = tf.Print(A1,[A1],message="A1")
    # Evaluating p(x|z)
    x_min = tf.reduce_min(x)
    x_max = tf.reduce_max(x)
    z=tf.Print(z,[z],message="z")
    z = (z+1)*(x_max-x_min)/2+x_min
    means = tf.matmul(tf.ones([A1.get_shape().as_list()[0],z.get_shape().as_list()[0],latent_dim])*z,tf.transpose(A1, perm=[0,2,1]))+tf.matmul(tf.ones([B1.get_shape().as_list()[0],c.get_shape().as_list()[0],inp_cov_dim])*C_embed(c),tf.transpose(B1, perm=[0,2,1])) # (N,100) (n_samples,5000,100)
    prec = tf.square(DELTA_inv1)
    means = means+prec
    #means = tf.Print(means,[means],message="means")
    assert_op = tf.Assert(tf.greater(tf.reduce_min(means),0),[means])
    with tf.control_dependencies([assert_op]):
        x_post_prob_log = x*tf.log(means+1e-5)-means-tf.lgamma(x+1)
        x_post_prob_log = tf.reduce_mean(x_post_prob_log, axis=2, keep_dims=False) # wrt to inp dim
        x_post_prob_log = tf.reduce_mean(x_post_prob_log, axis=0, keep_dims=False) # wrt to theta
        x_post_prob_log = tf.reduce_mean(x_post_prob_log, axis=0, keep_dims=False) # wrt to q(x)
    #x_post_prob_log = tf.Print(x_post_prob_log,[x_post_prob_log], message="x_post_prob_log")
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

    train(z, closs, label_acc_adv_theta)
