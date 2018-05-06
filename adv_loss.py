import tensorflow as tf
import numpy as np
from discriminators import *

slim = tf.contrib.slim
ds = tf.contrib.distributions
st = tf.contrib.bayesflow.stochastic_tensor
graph_replace = tf.contrib.graph_editor.graph_replace

thresh_adv = 0.5

def generate_classifier(latent_dim, n_samples,keep_prob,reuse=False):
    with tf.variable_scope("generate_classifier",reuse=reuse):
        w = tf.random_normal([n_samples, latent_dim//2], mean=0, stddev=1.0)
        out = slim.fully_connected(w,32,activation_fn=tf.nn.elu, weights_regularizer=slim.l2_regularizer(0.5)) # (1,1024)
        # Sample Classifier

        t = tf.constant(0.001, dtype=tf.float32, shape=[n_samples,1])
        logits_gumbel = slim.fully_connected(out,64,activation_fn=tf.nn.elu,weights_regularizer=slim.l2_regularizer(0.005))
        logits_gumbel = slim.fully_connected(logits_gumbel,2*latent_dim,activation_fn = None, weights_regularizer=slim.l2_regularizer(0.005))
        logits_gumbel = tf.reshape(logits_gumbel,[-1,latent_dim,2])
        M_gluten = gumbel_softmax(logits_gumbel,t)
        M_gluten = tf.squeeze(tf.slice(M_gluten,[0,0,0],[-1,-1,1]),axis=2)

        logits_gumbel = slim.fully_connected(out,64,activation_fn=tf.nn.elu,weights_regularizer=slim.l2_regularizer(0.005))
        logits_gumbel = slim.fully_connected(logits_gumbel,2*latent_dim,activation_fn = None, weights_regularizer=slim.l2_regularizer(0.005))
        logits_gumbel = tf.reshape(logits_gumbel,[-1,latent_dim,2])
        M_ibd = gumbel_softmax(logits_gumbel,t)
        M_ibd = tf.squeeze(tf.slice(M_ibd,[0,0,0],[-1,-1,1]),axis=2)

        h = slim.fully_connected(out,64,activation_fn=tf.nn.elu,weights_regularizer=slim.l2_regularizer(1.0))
        h = tf.nn.dropout(h,keep_prob)
        W_gluten = slim.fully_connected(out,latent_dim,activation_fn = None,weights_regularizer=slim.l2_regularizer(1.0))
        h = slim.fully_connected(out,64,activation_fn=tf.nn.elu,weights_regularizer=slim.l2_regularizer(1.0))
        h = tf.nn.dropout(h,keep_prob)
        W_ibd = slim.fully_connected(out,latent_dim,activation_fn = None,weights_regularizer=slim.l2_regularizer(1.0))
    return M_gluten,M_ibd,W_gluten,W_ibd
def generator(inp_data_dim, inp_cov_dim, latent_dim, rank,keep_prob, n_samples=1, noise_dim=100, reuse=False):
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
        u = slim.fully_connected(out, 256, activation_fn=lrelu, weights_regularizer=slim.l2_regularizer(0.005), weights_initializer=tf.orthogonal_initializer(gain=2.0))
        u = slim.fully_connected(u,inp_data_dim*rank*2,activation_fn=None,weights_regularizer=slim.l2_regularizer(0.005))
        U_gumbel = tf.reshape(u, [-1,2,inp_data_dim,rank])

        v = slim.fully_connected(out, 256, activation_fn=lrelu, weights_regularizer=slim.l2_regularizer(0.005), weights_initializer=tf.orthogonal_initializer(gain=2.0))
        v = slim.fully_connected(v,2*latent_dim*rank,activation_fn=None,weights_regularizer=slim.l2_regularizer(0.005))
        V_gumbel = tf.reshape(v,[-1,2,rank,latent_dim])

        #t = slim.fully_connected(out, 256, activation_fn=lrelu, weights_regularizer=slim.l2_regularizer(0.001), weights_initializer=tf.orthogonal_initializer(gain=1.0))
        #t = slim.fully_connected(t,1,activation_fn=tf.sigmoid,weights_regularizer=slim.l2_regularizer(0.001), weights_initializer=tf.orthogonal_initializer(gain=1.0))
        #t = tf.Print(t,[t],message="Temperature...")
        #variable_summaries(t, name="Temperature")
        #t = tf.reshape(t,[-1,1,1])
        t = tf.constant(0.1, dtype=tf.float32,shape=[n_samples,1,1])
        logits_gumbel = tf.transpose(tf.matmul(U_gumbel,V_gumbel),perm=[0,2,3,1]) #(nsamples,inp_data_dim,latent_dim,2)
        #logits_gumbel = tf.Print(logits_gumbel,[logits_gumbel],message="logits_gumbel")
        variable_summaries(logits_gumbel,name="logits_gumbel")
        M = gumbel_softmax(logits_gumbel, t)
        M = tf.squeeze(tf.slice(M, [0,0,0,0],[-1,-1,-1,1]), axis=3)
       # M = tf.transpose(M,perm=[0,2,1])
        #M = tf.Print(M,[M],message="M output generator")
        #M = cloglog(logits_gumbel)
        variable_summaries(M,name="M_generator")
        M_gluten,M_ibd,W_gluten,W_ibd=generate_classifier(latent_dim,n_samples,keep_prob,reuse=reuse)
    return U,V,B,del_sample,M,M_gluten,M_ibd,W_gluten,W_ibd

def cal_theta_adv_loss(q_samples_A, q_samples_B, q_samples_D, inp_data_dim, inp_cov_dim, latent_dim, rank,keep_prob, n_samples = 100):
    # n_samples = 100
    p_samples_U = tf.random_normal([n_samples, inp_data_dim, rank])
    p_samples_V = tf.random_normal([n_samples, rank, latent_dim])
    p_samples_B = tf.random_normal([n_samples, inp_data_dim, inp_cov_dim])
    p_samples_D = tf.random_normal([n_samples, inp_data_dim])
    p_samples_W_cnp = tf.random_normal([n_samples, latent_dim])
    p_samples_W_diabetes = tf.random_normal([n_samples, latent_dim])
    p_samples_W_gluten = tf.random_normal([n_samples, latent_dim])
    p_samples_W_ibd = tf.random_normal([n_samples, latent_dim])
    p_samples_W_lactose = tf.random_normal([n_samples, latent_dim])
    p_samples_W_quino = tf.random_normal([n_samples, latent_dim])
    p_samples_M = gumbel_softmax(tf.ones([n_samples, inp_data_dim, latent_dim,2]), tf.constant(0.01, dtype=tf.float32,shape=(n_samples,1,1)))
    p_samples_Mcnp = gumbel_softmax(tf.ones([n_samples, latent_dim,2]), tf.constant(0.01, dtype=tf.float32,shape=(n_samples,1)))
    p_samples_Mcnp = tf.squeeze(tf.slice(p_samples_Mcnp,[0,0,0],[-1,-1,1]),axis=2)
    p_samples_Mdiabetes = gumbel_softmax(tf.ones([n_samples, latent_dim,2]), tf.constant(0.01, dtype=tf.float32,shape=(n_samples,1)))
    p_samples_Mdiabetes = tf.squeeze(tf.slice(p_samples_Mdiabetes,[0,0,0],[-1,-1,1]),axis=2)
    p_samples_Mgluten = gumbel_softmax(tf.ones([n_samples, latent_dim,2]), tf.constant(0.01, dtype=tf.float32,shape=(n_samples,1)))
    p_samples_Mgluten = tf.squeeze(tf.slice(p_samples_Mgluten,[0,0,0],[-1,-1,1]),axis=2)
    p_samples_Mibd = gumbel_softmax(tf.ones([n_samples, latent_dim,2]), tf.constant(0.01, dtype=tf.float32,shape=(n_samples,1)))
    p_samples_Mibd = tf.squeeze(tf.slice(p_samples_Mibd,[0,0,0],[-1,-1,1]),axis=2)
    p_samples_Mlactose = gumbel_softmax(tf.ones([n_samples, latent_dim,2]), tf.constant(0.01, dtype=tf.float32,shape=(n_samples,1)))
    p_samples_Mlactose = tf.squeeze(tf.slice(p_samples_Mlactose,[0,0,0],[-1,-1,1]),axis=2)
    p_samples_Mquino = gumbel_softmax(tf.ones([n_samples, latent_dim,2]), tf.constant(0.01, dtype=tf.float32,shape=(n_samples,1)))
    p_samples_Mquino = tf.squeeze(tf.slice(p_samples_Mquino,[0,0,0],[-1,-1,1]),axis=2)
    p_samples_M = tf.squeeze(tf.slice(p_samples_M,[0,0,0,0],[-1,-1,-1,1]),axis=3)
    q_samples_U, q_samples_V, q_samples_B, q_samples_D, q_samples_M, q_samples_Mgluten,q_samples_Mibd,q_samples_W_gluten,q_samples_W_ibd = generator(inp_data_dim,inp_cov_dim,latent_dim,rank,keep_prob,n_samples=n_samples, reuse=True)

    variable_summaries(p_samples_M, name="p_samples_M")
    variable_summaries(q_samples_M, name="q_samples_M")
    q_ratio_m = 0

    p_ratio = U_ratio(p_samples_U,inp_data_dim,rank)
    q_ratio = U_ratio(q_samples_U,inp_data_dim,rank, reuse=True)
    d_loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=p_ratio, labels=tf.ones_like(p_ratio)))
    d_loss_i = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=q_ratio, labels=tf.zeros_like(q_ratio)))
    correct_labels_adv = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.greater(tf.sigmoid(p_ratio),thresh_adv), tf.int32),1),tf.float32)) + tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.less_equal(tf.sigmoid(q_ratio),thresh_adv), tf.int32),1),tf.float32))
    label_acc_adv_u = correct_labels_adv/(2*n_samples)
    label_acc_adv_u = tf.Print(label_acc_adv_u, [label_acc_adv_u], message="label_acc_adv_u")
    dloss_u = d_loss_d+d_loss_i
    q_ratio_m += p_ratio

    p_ratio = V_ratio(p_samples_V,latent_dim,rank)
    q_ratio = V_ratio(q_samples_V,latent_dim,rank, reuse=True)
    d_loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=p_ratio, labels=tf.ones_like(p_ratio)))
    d_loss_i = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=q_ratio, labels=tf.zeros_like(q_ratio)))
    correct_labels_adv = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.greater(tf.sigmoid(p_ratio),thresh_adv), tf.int32),1),tf.float32)) + tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.less_equal(tf.sigmoid(q_ratio),thresh_adv), tf.int32),1),tf.float32))
    print("correct_labels_shape:",p_ratio.get_shape().as_list())
    label_acc_adv_v = correct_labels_adv/(2*n_samples)
    dloss_v = d_loss_d+d_loss_i
    label_acc_adv_v = tf.Print(label_acc_adv_v, [label_acc_adv_v], message="label_acc_adv_v")
    q_ratio_m += p_ratio

    p_ratio = B_ratio(p_samples_B,inp_data_dim,inp_cov_dim)
    q_ratio = B_ratio(q_samples_B,inp_data_dim,inp_cov_dim, reuse=True)
    d_loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=p_ratio, labels=tf.ones_like(p_ratio)))
    d_loss_i = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=q_ratio, labels=tf.zeros_like(q_ratio)))
    correct_labels_adv = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.greater(tf.sigmoid(p_ratio),thresh_adv), tf.int32),1),tf.float32)) + tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.less_equal(tf.sigmoid(q_ratio),thresh_adv), tf.int32),1),tf.float32))
    label_acc_adv_b = correct_labels_adv/(2*n_samples)
    label_acc_adv_b = tf.Print(label_acc_adv_b, [label_acc_adv_b], message="label_acc_adv_b")
    dloss_b = d_loss_d+d_loss_i
    q_ratio_m += p_ratio

    p_ratio = D_ratio(p_samples_D)
    q_ratio = D_ratio(q_samples_D, reuse=True)
    d_loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=p_ratio, labels=tf.ones_like(p_ratio)))
    d_loss_i = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=q_ratio, labels=tf.zeros_like(q_ratio)))
    correct_labels_adv = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.greater(tf.sigmoid(p_ratio),thresh_adv), tf.int32),1),tf.float32)) + tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.less_equal(tf.sigmoid(q_ratio),thresh_adv), tf.int32),1),tf.float32))
    label_acc_adv_d = correct_labels_adv/(2*n_samples)
    label_acc_adv_d = tf.Print(label_acc_adv_d, [label_acc_adv_d], message="label_acc_adv_d")
    dloss_d = d_loss_d+d_loss_i
    q_ratio_m += p_ratio

    p_ratio = M_ratio(p_samples_M,inp_data_dim, latent_dim)
    q_ratio = M_ratio(q_samples_M,inp_data_dim, latent_dim, reuse=True)
    d_loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=p_ratio, labels=tf.ones_like(p_ratio)))
    d_loss_i = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=q_ratio, labels=tf.zeros_like(q_ratio)))
    correct_labels_adv = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.greater(tf.sigmoid(p_ratio),thresh_adv), tf.int32),1),tf.float32)) + tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.less_equal(tf.sigmoid(q_ratio),thresh_adv), tf.int32),1),tf.float32))
    label_acc_adv_m = correct_labels_adv/(2*n_samples)
    label_acc_adv_m = tf.Print(label_acc_adv_m, [label_acc_adv_m], message="label_acc_adv_m")
    dloss_m = d_loss_d+d_loss_i
    q_ratio_m += p_ratio

    p_ratio = Mgluten_ratio(p_samples_Mgluten)
    q_ratio = Mgluten_ratio(q_samples_Mgluten, reuse=True)
    d_loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=p_ratio, labels=tf.ones_like(p_ratio)))
    d_loss_i = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=q_ratio, labels=tf.zeros_like(q_ratio)))
    correct_labels_adv = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.greater(tf.sigmoid(p_ratio),thresh_adv), tf.int32),1),tf.float32)) + tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.less_equal(tf.sigmoid(q_ratio),thresh_adv), tf.int32),1),tf.float32))
    label_acc_adv_mgluten = correct_labels_adv/(2*n_samples)
    label_acc_adv_mgluten = tf.Print(label_acc_adv_mgluten, [label_acc_adv_mgluten], message="label_acc_adv_mgluten")
    dloss_mgluten = d_loss_d+d_loss_i
    q_ratio_m += p_ratio

    p_ratio = Mibd_ratio(p_samples_Mibd)
    q_ratio = Mibd_ratio(q_samples_Mibd, reuse=True)
    d_loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=p_ratio, labels=tf.ones_like(p_ratio)))
    d_loss_i = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=q_ratio, labels=tf.zeros_like(q_ratio)))
    correct_labels_adv = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.greater(tf.sigmoid(p_ratio),thresh_adv), tf.int32),1),tf.float32)) + tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.less_equal(tf.sigmoid(q_ratio),thresh_adv), tf.int32),1),tf.float32))
    label_acc_adv_mibd = correct_labels_adv/(2*n_samples)
    label_acc_adv_mibd = tf.Print(label_acc_adv_mibd, [label_acc_adv_mibd], message="label_acc_adv_mibd")
    dloss_mibd = d_loss_d+d_loss_i
    q_ratio_m += p_ratio


    p_ratio = Wgluten_ratio(p_samples_W_gluten)
    q_ratio = Wgluten_ratio(q_samples_W_gluten, reuse=True)
    d_loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=p_ratio, labels=tf.ones_like(p_ratio)))
    d_loss_i = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=q_ratio, labels=tf.zeros_like(q_ratio)))
    correct_labels_adv = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.greater(tf.sigmoid(p_ratio),thresh_adv), tf.int32),1),tf.float32)) + tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.less_equal(tf.sigmoid(q_ratio),thresh_adv), tf.int32),1),tf.float32))
    label_acc_adv_wgluten = correct_labels_adv/(2*n_samples)
    label_acc_adv_wgluten = tf.Print(label_acc_adv_wgluten, [label_acc_adv_wgluten], message="label_acc_adv_wgluten")
    dloss_wgluten = d_loss_d+d_loss_i
    q_ratio_m += p_ratio

    p_ratio = Wibd_ratio(p_samples_W_ibd)
    q_ratio = Wibd_ratio(q_samples_W_ibd, reuse=True)
    d_loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=p_ratio, labels=tf.ones_like(p_ratio)))
    d_loss_i = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=q_ratio, labels=tf.zeros_like(q_ratio)))
    correct_labels_adv = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.greater(tf.sigmoid(p_ratio),thresh_adv), tf.int32),1),tf.float32)) + tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.less_equal(tf.sigmoid(q_ratio),thresh_adv), tf.int32),1),tf.float32))
    label_acc_adv_wibd = correct_labels_adv/(2*n_samples)
    label_acc_adv_wibd = tf.Print(label_acc_adv_wibd, [label_acc_adv_wibd], message="label_acc_adv_wibd")
    dloss_wibd = d_loss_d+d_loss_i
    q_ratio_m += p_ratio


    dloss = dloss_u+dloss_v+dloss_b+dloss_d+dloss_m+30*(dloss_mgluten+dloss_mibd)+dloss_wgluten+dloss_wibd
    #Adversary Accuracy
    # correct_labels_adv = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.greater(tf.sigmoid(p_ratio),thresh_adv), tf.int32),1),tf.float32)) + tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.less_equal(tf.sigmoid(q_ratio),thresh_adv), tf.int32),0),tf.float32))
    label_acc_adv_theta = (label_acc_adv_b+label_acc_adv_d+label_acc_adv_u+label_acc_adv_v+label_acc_adv_m+label_acc_adv_mgluten+label_acc_adv_mibd+label_acc_adv_wgluten+label_acc_adv_wibd)/17.0
    label_acc_adv_theta = tf.Print(label_acc_adv_theta, [label_acc_adv_theta], message="label_acc_adv_theta")
    return dloss, label_acc_adv_theta, q_ratio_m


def countN1(out_prob, orig, name):
    pred = np.argmax(out_prob,axis=1)
    n1 = (pred==1).nonzero()[0].shape[0]
    n0 = (pred==0).nonzero()[0].shape[0]
    print("Pred: For ",name," #1:",n1," #0:",n0)
    n1 = (orig==1).nonzero()[0].shape[0]
    n0 = (orig==0).nonzero()[0].shape[0]
    print("Input: For ",name," #1:",n1," #0:",n0)


def classifier_cnp(x_input,labels,W,M, reuse=False):
    with tf.variable_scope("Classifier_cnp", reuse=reuse):
        #x_input += tf.random_normal(x_input.get_shape().as_list())
        b = tf.get_variable("bias",shape=[1],dtype=tf.float32)
        W = W*M
        labels = tf.cast(labels,tf.float32)
        logits = tf.matmul(x_input,W,transpose_b=True)+b
        probs = tf.sigmoid(logits) # (N,n_samples)
        loss = tf.expand_dims(labels,axis=1)*tf.log(probs+1e-5) + tf.expand_dims(1-labels,axis=1)*tf.log(1-probs+1e-5)
        cl_loss = tf.reduce_mean(loss,axis=1,keep_dims=False)
        cl_loss = -tf.reduce_mean(cl_loss)
    return tf.reduce_mean(tf.stack([1-probs,probs],axis=2),axis=1,keep_dims=False), cl_loss

def classifier_diabetes(x_input,labels,W,M, reuse=False):
    with tf.variable_scope("Classifier_diabetes", reuse=reuse):
        #x_input += tf.random_normal(x_input.get_shape().as_list())
        b = tf.get_variable("bias",shape=[1],dtype=tf.float32)
        W = W*M
        labels = tf.cast(labels,tf.float32)
        logits = tf.matmul(x_input,W,transpose_b=True)+b
        probs = tf.sigmoid(logits) # (N,n_samples)
        loss = tf.expand_dims(labels,axis=1)*tf.log(probs+1e-5) + tf.expand_dims(1-labels,axis=1)*tf.log(1-probs+1e-5)
        cl_loss = tf.reduce_mean(loss,axis=1,keep_dims=False)
        cl_loss = -tf.reduce_mean(cl_loss)
    return tf.reduce_mean(tf.stack([1-probs,probs],axis=2),axis=1,keep_dims=False), cl_loss

def classifier_gluten(x_input,labels, W,M,reuse=False):
    with tf.variable_scope("Classifier_gluten", reuse=reuse):
        #x_input += tf.random_normal(x_input.get_shape().as_list())
        b = tf.get_variable("bias",shape=[1],dtype=tf.float32)
        W = W*M
        labels = tf.cast(labels,tf.float32)
        logits = tf.matmul(x_input,W,transpose_b=True)+b
        probs = tf.sigmoid(logits) # (N,n_samples)
        loss = tf.expand_dims(labels,axis=1)*tf.log(probs+1e-5) + tf.expand_dims(1-labels,axis=1)*tf.log(1-probs+1e-5)
        cl_loss = tf.reduce_mean(loss,axis=1,keep_dims=False)
        cl_loss = -tf.reduce_mean(cl_loss)
    return tf.reduce_mean(tf.stack([1-probs,probs],axis=2),axis=1,keep_dims=False), cl_loss

def classifier_ibd(x_input,labels, W,M,reuse=False):
    with tf.variable_scope("Classifier_ibd", reuse=reuse):
        #x_input += tf.random_normal(x_input.get_shape().as_list())
        b = tf.get_variable("bias",shape=[1],dtype=tf.float32)
        W = W*M
        labels = tf.cast(labels,tf.float32)
        logits = tf.matmul(x_input,W,transpose_b=True)+b
        probs = tf.sigmoid(logits) # (N,n_samples)
        loss = tf.expand_dims(labels,axis=1)*tf.log(probs+1e-5) + tf.expand_dims(1-labels,axis=1)*tf.log(1-probs+1e-5)
        cl_loss = tf.reduce_mean(loss,axis=1,keep_dims=False)
        cl_loss = -tf.reduce_mean(cl_loss)
    return tf.reduce_mean(tf.stack([1-probs,probs],axis=2),axis=1,keep_dims=False), cl_loss

def classifier_lactose(x_input,labels, W,M,reuse=False):
    with tf.variable_scope("Classifier_lactose", reuse=reuse):
        #x_input += tf.random_normal(x_input.get_shape().as_list())
        b = tf.get_variable("bias",shape=[1],dtype=tf.float32)
        W = W*M
        labels = tf.cast(labels,tf.float32)
        logits = tf.matmul(x_input,W,transpose_b=True)+b
        probs = tf.sigmoid(logits) # (N,n_samples)
        loss = tf.expand_dims(labels,axis=1)*tf.log(probs+1e-5) + tf.expand_dims(1-labels,axis=1)*tf.log(1-probs+1e-5)
        cl_loss = tf.reduce_mean(loss,axis=1,keep_dims=False)
        cl_loss = -tf.reduce_mean(cl_loss)
    return tf.reduce_mean(tf.stack([1-probs,probs],axis=2),axis=1,keep_dims=False), cl_loss

def classifier_quino(x_input,labels, W,M,reuse=False):
    with tf.variable_scope("Classifier_quino", reuse=reuse):
        #x_input += tf.random_normal(x_input.get_shape().as_list())
        b = tf.get_variable("bias",shape=[1],dtype=tf.float32)
        W = W*M
        labels = tf.cast(labels,tf.float32)
        logits = tf.matmul(x_input,W,transpose_b=True)+b
        probs = tf.sigmoid(logits) # (N,n_samples)
        loss = tf.expand_dims(labels,axis=1)*tf.log(probs+1e-5) + tf.expand_dims(1-labels,axis=1)*tf.log(1-probs+1e-5)
        cl_loss = tf.reduce_mean(loss,axis=1,keep_dims=False)
        cl_loss = -tf.reduce_mean(cl_loss)
    return tf.reduce_mean(tf.stack([1-probs,probs],axis=2),axis=1,keep_dims=False), cl_loss

