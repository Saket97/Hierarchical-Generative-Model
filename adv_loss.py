import tensorflow as tf
import numpy as np
from discriminators import *

slim = tf.contrib.slim
ds = tf.contrib.distributions
st = tf.contrib.bayesflow.stochastic_tensor
graph_replace = tf.contrib.graph_editor.graph_replace

thresh_adv = 0.5

def generator(inp_data_dim, latent_dim, rank, n_samples=1, noise_dim=100, reuse=False):
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
    return U,V,del_sample,M

def cal_theta_adv_loss(q_samples_A, q_samples_D, inp_data_dim, latent_dim, rank, n_samples = 100):
    # n_samples = 100
    p_samples_U = tf.random_normal([n_samples, inp_data_dim, rank])
    p_samples_V = tf.random_normal([n_samples, rank, latent_dim])
    p_samples_D = tf.random_normal([n_samples, inp_data_dim])
    p_samples_M = gumbel_softmax(tf.ones([n_samples, inp_data_dim, latent_dim,2]), tf.constant(0.01, dtype=tf.float32,shape=(n_samples,1,1)))
    p_samples_M = tf.squeeze(tf.slice(p_samples_M,[0,0,0,0],[-1,-1,-1,1]),axis=3)
    q_samples_U, q_samples_V, q_samples_D, q_samples_M = generator(inp_data_dim,latent_dim,rank,n_samples=n_samples, reuse=True)

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
    q_ratio_m += tf.reduce_mean(q_ratio)

    p_ratio = V_ratio(p_samples_V,latent_dim,rank)
    q_ratio = V_ratio(q_samples_V,latent_dim,rank, reuse=True)
    d_loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=p_ratio, labels=tf.ones_like(p_ratio)))
    d_loss_i = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=q_ratio, labels=tf.zeros_like(q_ratio)))
    correct_labels_adv = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.greater(tf.sigmoid(p_ratio),thresh_adv), tf.int32),1),tf.float32)) + tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.less_equal(tf.sigmoid(q_ratio),thresh_adv), tf.int32),1),tf.float32))
    print("correct_labels_shape:",p_ratio.get_shape().as_list())
    label_acc_adv_v = correct_labels_adv/(2*n_samples)
    dloss_v = d_loss_d+d_loss_i
    label_acc_adv_v = tf.Print(label_acc_adv_v, [label_acc_adv_v], message="label_acc_adv_v")
    q_ratio_m += tf.reduce_mean(q_ratio)
    #dloss_v = 0
    #label_acc_adv_v = 1

    p_ratio = D_ratio(p_samples_D)
    q_ratio = D_ratio(q_samples_D, reuse=True)
    d_loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=p_ratio, labels=tf.ones_like(p_ratio)))
    d_loss_i = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=q_ratio, labels=tf.zeros_like(q_ratio)))
    correct_labels_adv = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.greater(tf.sigmoid(p_ratio),thresh_adv), tf.int32),1),tf.float32)) + tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.less_equal(tf.sigmoid(q_ratio),thresh_adv), tf.int32),1),tf.float32))
    label_acc_adv_d = correct_labels_adv/(2*n_samples)
    label_acc_adv_d = tf.Print(label_acc_adv_d, [label_acc_adv_d], message="label_acc_adv_d")
    dloss_d = d_loss_d+d_loss_i
    q_ratio_m += tf.reduce_mean(q_ratio)

    p_ratio = M_ratio(p_samples_M,inp_data_dim, latent_dim)
    q_ratio = M_ratio(q_samples_M,inp_data_dim, latent_dim, reuse=True)
    d_loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=p_ratio, labels=tf.ones_like(p_ratio)))
    d_loss_i = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=q_ratio, labels=tf.zeros_like(q_ratio)))
    correct_labels_adv = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.greater(tf.sigmoid(p_ratio),thresh_adv), tf.int32),1),tf.float32)) + tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.less_equal(tf.sigmoid(q_ratio),thresh_adv), tf.int32),1),tf.float32))
    label_acc_adv_m = correct_labels_adv/(2*n_samples)
    label_acc_adv_m = tf.Print(label_acc_adv_m, [label_acc_adv_m], message="label_acc_adv_m")
    dloss_m = d_loss_d+d_loss_i
    q_ratio_m += tf.reduce_mean(q_ratio)

    dloss = dloss_u+dloss_v+dloss_d+dloss_m
    
    #Adversary Accuracy
    # correct_labels_adv = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.greater(tf.sigmoid(p_ratio),thresh_adv), tf.int32),1),tf.float32)) + tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.less_equal(tf.sigmoid(q_ratio),thresh_adv), tf.int32),0),tf.float32))
    label_acc_adv_theta = (label_acc_adv_d+label_acc_adv_u+label_acc_adv_v+label_acc_adv_m)/4.0
    label_acc_adv_theta = tf.Print(label_acc_adv_theta, [label_acc_adv_theta], message="label_acc_adv_theta")   
    return dloss, label_acc_adv_theta, q_ratio_m
