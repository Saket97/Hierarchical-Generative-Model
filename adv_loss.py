import tensorflow as tf
import numpy as np
from discriminators import *

slim = tf.contrib.slim
ds = tf.contrib.distributions
st = tf.contrib.bayesflow.stochastic_tensor
graph_replace = tf.contrib.graph_editor.graph_replace

thresh_adv = 0.5

def generate_classifier(latent_dim, n_samples,reuse=False):
    with tf.variable_scope("generate_classifier",reuse=reuse):
        w = tf.random_normal([n_samples, latent_dim//2], mean=0, stddev=1.0)
        out = slim.fully_connected(w,32,activation_fn=tf.nn.elu, weights_regularizer=slim.l2_regularizer(0.05)) # (1,1024)
        # Sample Classifier
        logits_gumbel = slim.fully_connected(out,64,activation_fn=tf.nn.elu,weights_regularizer=slim.l2_regularizer(0.005))
        logits_gumbel = slim.fully_connected(logits_gumbel,2*latent_dim,activation_fn = None, weights_regularizer=slim.l2_regularizer(0.005))
        logits_gumbel = tf.reshape(logits_gumbel,[-1,latent_dim,2])
        #logits_gumbel = tf.Print(logits_gumbel,[logits_gumbel],message="logits gumbel tb")
        variable_summaries(logits_gumbel,name="logits_gumbel Mtb")
        t = tf.constant(0.001, dtype=tf.float32, shape=[n_samples,1])
        M_tb = gumbel_softmax(logits_gumbel,t)
        M_tb = tf.squeeze(tf.slice(M_tb,[0,0,0],[-1,-1,1]),axis=2)

        logits_gumbel = slim.fully_connected(out,64,activation_fn=tf.nn.elu,weights_regularizer=slim.l2_regularizer(0.005))
        logits_gumbel = slim.fully_connected(logits_gumbel,2*latent_dim,activation_fn = None, weights_regularizer=slim.l2_regularizer(0.005))
        logits_gumbel = tf.reshape(logits_gumbel,[-1,latent_dim,2])
        M_active = gumbel_softmax(logits_gumbel,t)
        M_active = tf.squeeze(tf.slice(M_active,[0,0,0],[-1,-1,1]),axis=2)

        logits_gumbel = slim.fully_connected(out,64,activation_fn=tf.nn.elu,weights_regularizer=slim.l2_regularizer(0.005))
        logits_gumbel = slim.fully_connected(logits_gumbel,2*latent_dim,activation_fn = None, weights_regularizer=slim.l2_regularizer(0.005))
        logits_gumbel = tf.reshape(logits_gumbel,[-1,latent_dim,2])
        M_latent = gumbel_softmax(logits_gumbel,t)
        M_latent = tf.squeeze(tf.slice(M_latent,[0,0,0],[-1,-1,1]),axis=2)

        logits_gumbel = slim.fully_connected(out,64,activation_fn=tf.nn.elu,weights_regularizer=slim.l2_regularizer(0.005))
        logits_gumbel = slim.fully_connected(logits_gumbel,2*latent_dim,activation_fn = None, weights_regularizer=slim.l2_regularizer(0.005))
        logits_gumbel = tf.reshape(logits_gumbel,[-1,latent_dim,2])
        M_hiv = gumbel_softmax(logits_gumbel,t)
        M_hiv = tf.squeeze(tf.slice(M_hiv,[0,0,0],[-1,-1,1]),axis=2)

        h = slim.fully_connected(out,64,activation_fn=tf.nn.elu)
        W_tb = slim.fully_connected(h,latent_dim,activation_fn = None)
        h = slim.fully_connected(out,64,activation_fn=tf.nn.elu)
        W_active = slim.fully_connected(h,latent_dim,activation_fn = None)
        h = slim.fully_connected(out,64,activation_fn=tf.nn.elu)
        W_latent = slim.fully_connected(h,latent_dim,activation_fn = None)
        h = slim.fully_connected(out,64,activation_fn=tf.nn.elu)
        W_hiv = slim.fully_connected(h,latent_dim,activation_fn = None)
    return M_tb,M_active,M_latent,M_hiv,W_tb,W_active,W_latent,W_hiv

def generator(inp_data_dim, inp_cov_dim, latent_dim, rank, n_samples=1, noise_dim=100, reuse=False):
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
        M_tb,M_active,M_latent,M_hiv,W_tb,W_active,W_latent,W_hiv=generate_classifier(latent_dim,n_samples,reuse=reuse)
    return U,V,B,del_sample,M,M_tb,M_active,M_latent,M_hiv,W_tb,W_active,W_latent,W_hiv

def cal_theta_adv_loss(q_samples_A, q_samples_B, q_samples_D, inp_data_dim, inp_cov_dim, latent_dim, rank, n_samples = 100):
    # n_samples = 100
    p_samples_U = tf.random_normal([n_samples, inp_data_dim, rank])
    p_samples_V = tf.random_normal([n_samples, rank, latent_dim])
    p_samples_B = tf.random_normal([n_samples, inp_data_dim, inp_cov_dim])
    p_samples_D = tf.random_normal([n_samples, inp_data_dim])
    p_samples_W_tb = tf.random_normal([n_samples, latent_dim])
    p_samples_W_active = tf.random_normal([n_samples, latent_dim])
    p_samples_W_latent = tf.random_normal([n_samples, latent_dim])
    p_samples_W_hiv = tf.random_normal([n_samples, latent_dim])
    p_samples_M = gumbel_softmax(tf.ones([n_samples, inp_data_dim, latent_dim,2]), tf.constant(0.01, dtype=tf.float32,shape=(n_samples,1,1)))
    p_samples_Mtb = gumbel_softmax(tf.ones([n_samples, latent_dim,2]), tf.constant(0.01, dtype=tf.float32,shape=(n_samples,1)))
    p_samples_Mtb = tf.squeeze(tf.slice(p_samples_Mtb,[0,0,0],[-1,-1,1]),axis=2)
    p_samples_Mactive = gumbel_softmax(tf.ones([n_samples, latent_dim,2]), tf.constant(0.01, dtype=tf.float32,shape=(n_samples,1)))
    p_samples_Mactive = tf.squeeze(tf.slice(p_samples_Mactive,[0,0,0],[-1,-1,1]),axis=2)
    p_samples_Mlatent = gumbel_softmax(tf.ones([n_samples, latent_dim,2]), tf.constant(0.01, dtype=tf.float32,shape=(n_samples,1)))
    p_samples_Mlatent = tf.squeeze(tf.slice(p_samples_Mlatent,[0,0,0],[-1,-1,1]),axis=2)
    p_samples_Mhiv = gumbel_softmax(tf.ones([n_samples, latent_dim,2]), tf.constant(0.01, dtype=tf.float32,shape=(n_samples,1)))
    p_samples_Mhiv = tf.squeeze(tf.slice(p_samples_Mhiv,[0,0,0],[-1,-1,1]),axis=2)
    p_samples_M = tf.squeeze(tf.slice(p_samples_M,[0,0,0,0],[-1,-1,-1,1]),axis=3)
    q_samples_U, q_samples_V, q_samples_B, q_samples_D, q_samples_M, q_samples_Mtb, q_samples_Mactive,q_samples_Mlatent,q_samples_Mhiv,q_samples_W_tb,q_samples_W_active,q_samples_W_latent,q_samples_W_hiv = generator(inp_data_dim,inp_cov_dim,latent_dim,rank,n_samples=n_samples, reuse=True)

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

    p_ratio = B_ratio(p_samples_B,inp_data_dim,inp_cov_dim)
    q_ratio = B_ratio(q_samples_B,inp_data_dim,inp_cov_dim, reuse=True)
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

    p_ratio = M_ratio(p_samples_M,inp_data_dim, latent_dim)
    q_ratio = M_ratio(q_samples_M,inp_data_dim, latent_dim, reuse=True)
    d_loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=p_ratio, labels=tf.ones_like(p_ratio)))
    d_loss_i = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=q_ratio, labels=tf.zeros_like(q_ratio)))
    correct_labels_adv = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.greater(tf.sigmoid(p_ratio),thresh_adv), tf.int32),1),tf.float32)) + tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.less_equal(tf.sigmoid(q_ratio),thresh_adv), tf.int32),1),tf.float32))
    label_acc_adv_m = correct_labels_adv/(2*n_samples)
    label_acc_adv_m = tf.Print(label_acc_adv_m, [label_acc_adv_m], message="label_acc_adv_m")
    dloss_m = d_loss_d+d_loss_i
    q_ratio_m += tf.reduce_mean(q_ratio)

    p_ratio = Mtb_ratio(p_samples_Mtb)
    q_ratio = Mtb_ratio(q_samples_Mtb, reuse=True)
    d_loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=p_ratio, labels=tf.ones_like(p_ratio)))
    d_loss_i = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=q_ratio, labels=tf.zeros_like(q_ratio)))
    correct_labels_adv = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.greater(tf.sigmoid(p_ratio),thresh_adv), tf.int32),1),tf.float32)) + tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.less_equal(tf.sigmoid(q_ratio),thresh_adv), tf.int32),1),tf.float32))
    label_acc_adv_mtb = correct_labels_adv/(2*n_samples)
    label_acc_adv_mtb = tf.Print(label_acc_adv_mtb, [label_acc_adv_mtb], message="label_acc_adv_mtb")
    dloss_mtb = d_loss_d+d_loss_i
    q_ratio_m += 40*tf.reduce_mean(q_ratio)

    p_ratio = Mactive_ratio(p_samples_Mactive)
    q_ratio = Mactive_ratio(q_samples_Mactive, reuse=True)
    d_loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=p_ratio, labels=tf.ones_like(p_ratio)))
    d_loss_i = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=q_ratio, labels=tf.zeros_like(q_ratio)))
    correct_labels_adv = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.greater(tf.sigmoid(p_ratio),thresh_adv), tf.int32),1),tf.float32)) + tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.less_equal(tf.sigmoid(q_ratio),thresh_adv), tf.int32),1),tf.float32))
    label_acc_adv_mactive = correct_labels_adv/(2*n_samples)
    label_acc_adv_mactive = tf.Print(label_acc_adv_mactive, [label_acc_adv_mactive], message="label_acc_adv_mactive")
    dloss_mactive = d_loss_d+d_loss_i
    q_ratio_m += tf.reduce_mean(q_ratio)

    p_ratio = Mlatent_ratio(p_samples_Mlatent)
    q_ratio = Mlatent_ratio(q_samples_Mlatent, reuse=True)
    d_loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=p_ratio, labels=tf.ones_like(p_ratio)))
    d_loss_i = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=q_ratio, labels=tf.zeros_like(q_ratio)))
    correct_labels_adv = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.greater(tf.sigmoid(p_ratio),thresh_adv), tf.int32),1),tf.float32)) + tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.less_equal(tf.sigmoid(q_ratio),thresh_adv), tf.int32),1),tf.float32))
    label_acc_adv_mlatent = correct_labels_adv/(2*n_samples)
    label_acc_adv_mlatent = tf.Print(label_acc_adv_mlatent, [label_acc_adv_mlatent], message="label_acc_adv_mlatent")
    dloss_mlatent = d_loss_d+d_loss_i
    q_ratio_m += tf.reduce_mean(q_ratio)

    p_ratio = Mhiv_ratio(p_samples_Mhiv)
    q_ratio = Mhiv_ratio(q_samples_Mhiv, reuse=True)
    d_loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=p_ratio, labels=tf.ones_like(p_ratio)))
    d_loss_i = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=q_ratio, labels=tf.zeros_like(q_ratio)))
    correct_labels_adv = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.greater(tf.sigmoid(p_ratio),thresh_adv), tf.int32),1),tf.float32)) + tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.less_equal(tf.sigmoid(q_ratio),thresh_adv), tf.int32),1),tf.float32))
    label_acc_adv_mhiv = correct_labels_adv/(2*n_samples)
    label_acc_adv_mhiv = tf.Print(label_acc_adv_mhiv, [label_acc_adv_mhiv], message="label_acc_adv_mhiv")
    dloss_mhiv = d_loss_d+d_loss_i
    q_ratio_m += tf.reduce_mean(q_ratio)

    p_ratio = Wtb_ratio(p_samples_W_tb)
    q_ratio = Wtb_ratio(q_samples_W_tb, reuse=True)
    d_loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=p_ratio, labels=tf.ones_like(p_ratio)))
    d_loss_i = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=q_ratio, labels=tf.zeros_like(q_ratio)))
    correct_labels_adv = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.greater(tf.sigmoid(p_ratio),thresh_adv), tf.int32),1),tf.float32)) + tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.less_equal(tf.sigmoid(q_ratio),thresh_adv), tf.int32),1),tf.float32))
    label_acc_adv_wtb = correct_labels_adv/(2*n_samples)
    label_acc_adv_wtb = tf.Print(label_acc_adv_wtb, [label_acc_adv_wtb], message="label_acc_adv_wtb")
    dloss_wtb = d_loss_d+d_loss_i
    q_ratio_m += tf.reduce_mean(q_ratio)

    p_ratio = Wactive_ratio(p_samples_W_active)
    q_ratio = Wactive_ratio(q_samples_W_active, reuse=True)
    d_loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=p_ratio, labels=tf.ones_like(p_ratio)))
    d_loss_i = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=q_ratio, labels=tf.zeros_like(q_ratio)))
    correct_labels_adv = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.greater(tf.sigmoid(p_ratio),thresh_adv), tf.int32),1),tf.float32)) + tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.less_equal(tf.sigmoid(q_ratio),thresh_adv), tf.int32),1),tf.float32))
    label_acc_adv_wactive = correct_labels_adv/(2*n_samples)
    label_acc_adv_wactive = tf.Print(label_acc_adv_wactive, [label_acc_adv_wactive], message="label_acc_adv_wactive")
    dloss_wactive = d_loss_d+d_loss_i
    q_ratio_m += tf.reduce_mean(q_ratio)

    p_ratio = Wlatent_ratio(p_samples_W_latent)
    q_ratio = Wlatent_ratio(q_samples_W_latent, reuse=True)
    d_loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=p_ratio, labels=tf.ones_like(p_ratio)))
    d_loss_i = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=q_ratio, labels=tf.zeros_like(q_ratio)))
    correct_labels_adv = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.greater(tf.sigmoid(p_ratio),thresh_adv), tf.int32),1),tf.float32)) + tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.less_equal(tf.sigmoid(q_ratio),thresh_adv), tf.int32),1),tf.float32))
    label_acc_adv_wlatent = correct_labels_adv/(2*n_samples)
    label_acc_adv_wlatent = tf.Print(label_acc_adv_wlatent, [label_acc_adv_wlatent], message="label_acc_adv_wlatent")
    dloss_wlatent = d_loss_d+d_loss_i
    q_ratio_m += tf.reduce_mean(q_ratio)

    p_ratio = Whiv_ratio(p_samples_W_hiv)
    q_ratio = Whiv_ratio(q_samples_W_hiv, reuse=True)
    d_loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=p_ratio, labels=tf.ones_like(p_ratio)))
    d_loss_i = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=q_ratio, labels=tf.zeros_like(q_ratio)))
    correct_labels_adv = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.greater(tf.sigmoid(p_ratio),thresh_adv), tf.int32),1),tf.float32)) + tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.less_equal(tf.sigmoid(q_ratio),thresh_adv), tf.int32),1),tf.float32))
    label_acc_adv_whiv = correct_labels_adv/(2*n_samples)
    label_acc_adv_whiv = tf.Print(label_acc_adv_whiv, [label_acc_adv_whiv], message="label_acc_adv_whiv")
    dloss_whiv = d_loss_d+d_loss_i
    q_ratio_m += tf.reduce_mean(q_ratio)

    dloss = dloss_u+dloss_v+dloss_b+dloss_d+dloss_m+30*(dloss_mtb+dloss_mactive+dloss_mlatent+dloss_mhiv)+dloss_wtb+dloss_wactive+dloss_wlatent+dloss_whiv
    
    #Adversary Accuracy
    # correct_labels_adv = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.greater(tf.sigmoid(p_ratio),thresh_adv), tf.int32),1),tf.float32)) + tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.less_equal(tf.sigmoid(q_ratio),thresh_adv), tf.int32),0),tf.float32))
    label_acc_adv_theta = (label_acc_adv_b+label_acc_adv_d+label_acc_adv_u+label_acc_adv_v+label_acc_adv_m+label_acc_adv_mtb+label_acc_adv_mactive+label_acc_adv_mlatent+label_acc_adv_mhiv+label_acc_adv_wtb+label_acc_adv_wactive+label_acc_adv_wlatent+label_acc_adv_whiv)/13.0
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

def classifier(x_input,labels,W,M, reuse=False):
    with tf.variable_scope("Classifier", reuse=reuse):
        b = tf.get_variable("bias",shape=[1],dtype=tf.float32)
        W = W*M
        logits = tf.matmul(x_input,W,transpose_b=True)+b # (N,latent_dim)(latent_dim,n_samples) = (N,n_samples)
        #logits = slim.fully_connected(x_input, num_hiv_classes, activation_fn=None, weights_regularizer=slim.l2_regularizer(0.1))
        probs = tf.sigmoid(logits) # (N,n_samples)
        loss = tf.expand_dims(tf.cast(labels,tf.float32),axis=1)*tf.log(probs+1e-5) + tf.expand_dims(1-tf.cast(labels,tf.float32),axis=1)*tf.log(1-probs+1e-5)
        #cl_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        cl_loss = tf.reduce_mean(loss,axis=1,keep_dims=False)
        cl_loss = -tf.reduce_mean(cl_loss)
    return tf.reduce_mean(tf.stack([1-probs,probs],axis=2),axis=1,keep_dims=False), cl_loss

def classifier_tb(x_input,labels,W,M, reuse=False):
    with tf.variable_scope("Classifier_tb", reuse=reuse):
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

def classifier_active_tb(x_input,labels,W,M, reuse=False):
    with tf.variable_scope("Classifier_active_tb", reuse=reuse):
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

def classifier_latent_tb(x_input,labels, W,M,reuse=False):
    with tf.variable_scope("Classifier_latent_tb", reuse=reuse):
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

