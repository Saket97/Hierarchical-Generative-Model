import tensorflow as tf
import numpy as np
import pickle as pkl
import math
import os
import sys
import argparse
import sklearn.metrics as skm
import sklearn.preprocessing as skp
from discriminators import *
from adv_loss import *

slim = tf.contrib.slim
ds = tf.contrib.distributions
st = tf.contrib.bayesflow.stochastic_tensor
graph_replace = tf.contrib.graph_editor.graph_replace

""" Hyperparameters """
latent_dim=40
nepoch = 1001
lr = 10**(-4)
batch_size = 10
ntrain = 30
test_batch_size = 20
ntest=20
inp_data_dim = 1000
eps_dim = 40
eps_nbasis=32
n_clf_epoch = 5000
thresh_adv = 0.5
rank = 30

""" tensorboard """
parser = argparse.ArgumentParser()
parser.add_argument('--logdir',type=str,default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),'tensorflow/mnist/logs/mnist_with_summaries'),help='Summaries log directory')
FLAGS, unparsed = parser.parse_known_args()
if tf.gfile.Exists(FLAGS.logdir):
    tf.gfile.DeleteRecursively(FLAGS.logdir)
tf.gfile.MakeDirs(FLAGS.logdir)

def load_dataset():
    raw_data = np.load("x.npy")
    global inp_data_dim
    inp_data_dim = raw_data.shape[1]
    m = np.mean(raw_data, axis=0)
    raw_data = (raw_data-m)/5.0
    #cov = (np.log10(cov+0.1))/5.0
    return raw_data[0:ntrain],raw_data[ntrain:ntrain+ntest]

X_train, X_test = load_dataset()
print("inp_data_dim:",inp_data_dim)
X_test = X_test.astype(np.float32)

def encoder(x, eps=None, n_layer=1, n_hidden=128, reuse=False):
    with tf.variable_scope("Encoder", reuse = reuse):
        h = slim.repeat(x, n_layer, slim.fully_connected, n_hidden, activation_fn=tf.nn.elu, weights_regularizer=slim.l2_regularizer(0.1))
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

def train(z, label_acc_adv_theta):
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith("data_net")]
    tr_vars = [var for var in t_vars if (var.name.startswith("U") or var.name.startswith("V") or var.name.startswith("B") or var.name.startswith("del") or var.name.startswith("M") or var.name.startswith("Mtb") or var.name.startswith("Mactive") or var.name.startswith("Mlatent") or var.name.startswith("Mhiv") or var.name.startswith("Wtb") or var.name.startswith("Wactive") or var.name.startswith("Wlatent") or var.name.startswith("Whiv"))]
    tp_var = [var for var in t_vars if var not in d_vars+c_vars+tr_vars]
    #print("tp_var:",tp_var)
    #print("tr_var:",tr_vars)
    assert(len(tp_var)+len(d_vars)+len(tr_vars) == len(t_vars))
    
    r_loss = tf.losses.get_regularization_loss()
    primal_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, use_locking=True, beta1=0.5)
    dual_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, use_locking=True, beta1=0.5)
    dual_optimizer_theta = tf.train.AdamOptimizer(learning_rate=1e-4, use_locking=True, beta1=0.5)
    #classifier_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3, use_locking=True)

    primal_grad = primal_optimizer.compute_gradients(primal_loss+r_loss, var_list=tp_var)
    capped_g_grad = []
    for grad,var in primal_grad:
        if grad is not None:
            capped_g_grad.append((tf.clip_by_value(grad,-0.1,0.1),var))
    primal_train_op = primal_optimizer.apply_gradients(capped_g_grad)

    adversary_grad = dual_optimizer.compute_gradients(dual_loss+r_loss,var_list=d_vars)
    capped_g_grad = []
    for grad,var in adversary_grad:
        if grad is not None:
            capped_g_grad.append((tf.clip_by_value(grad,-0.1,0.1),var))
    adversary_train_op = dual_optimizer.apply_gradients(capped_g_grad)
  
    #adversary_theta_train_op = dual_optimizer_theta.minimize(dual_loss_theta+r_loss, var_list=tr_vars)
    adversary_theta_grad = dual_optimizer_theta.compute_gradients(dual_loss_theta+r_loss, var_list=tr_vars)
    capped_g_grad = []
    for grad,var in adversary_theta_grad:
        if grad is not None:
            capped_g_grad.append((tf.clip_by_value(grad,-0.1,0.1),var))
    adversary_theta_train_op = dual_optimizer_theta.apply_gradients(capped_g_grad)

    train_op = tf.group(primal_train_op, adversary_train_op)
    
    # Test Set Graph
    eps = tf.random_normal(tf.stack([eps_nbasis, test_batch_size,eps_dim]))
    z_test, _, _ = encoder(X_test,eps,reuse=True)
    U_test,V_test,D_test = generator(inp_data_dim,latent_dim,rank,n_samples=500, reuse=True)
    A_old = tf.matmul(U_test,V_test)
    A = A_old*M_test
    means = tf.matmul(tf.ones([A.get_shape().as_list()[0],z_test.get_shape().as_list()[0],latent_dim])*z_test,tf.transpose(A, perm=[0,2,1]))
    prec = tf.square(D_test)
    t = (X_test-means)
    t1 = t*tf.expand_dims(prec, axis=1)*t
    t1 = -0.5*tf.reduce_sum(t1, axis=2)
    t2 = 0.5*tf.expand_dims(tf.reduce_sum(tf.log(1e-3+prec), axis=1), axis=1)
    t3 = -inp_data_dim*0.5*tf.log(2*math.pi)
    x_post_prob_log_test = t1+t2+t3
    #x_post_prob_log_test = tf.reduce_mean(x_post_prob_log_test, axis=1)
    x_post_prob_log_test = tf.reduce_mean(x_post_prob_log_test, axis=0) # expect wrt theta
        
    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.logdir, graph = tf.get_default_graph())
    sess = tf.Session()
    #saver.restore(sess,"/opt/data/saket/gene_data/model_spike/model.ckpt-249")
    sess.run(tf.global_variables_initializer())
    si_list = []
    vtp_list = []
    post_test_list = []
    dt_list = []
    test_lik_list1 = []
    for i in range(nepoch):
        for j in range(ntrain//batch_size):
            xmb = X_train[j*batch_size:(j+1)*batch_size]
            cmb = C_train[j*batch_size:(j+1)*batch_size]
            sess.run(train_op, feed_dict={x:xmb})
            for gen in range(1):
                sess.run(adversary_theta_train_op, feed_dict={x:xmb})
            vtp_loss = sess.run([primal_loss], feed_dict={x:xmb})
            vtp_list.append(vtp_loss)
        if i%100 == 0:
            Td_,KL_neg_r_q_,x_post_prob_log_,logz_,logr_,d_loss_d_,d_loss_i_,label_acc_adv_,label_acc_adv_theta_,dual_loss_theta_ = sess.run([Td,KL_neg_r_q,x_post_prob_log,logz,logr,d_loss_d,d_loss_i,label_acc_adv, label_acc_adv_theta, dual_loss_theta], feed_dict={x:xmb})
            print("epoch:",i," vtp:",vtp_list[-1], " x_post:",x_post_prob_log_)
        if i%500 == 0:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary = sess.run(merged,feed_dict={x:xmb})
            train_writer.add_run_metadata(run_metadata, 'step %i'%(i))
            train_writer.add_summary(summary, i)

        if i%100 == 0:
            test_lik_list = []
            for i in range(100):
                test_lik = sess.run(x_post_prob_log_test)
                test_lik_list.append(test_lik)
            test_lik = np.stack(test_lik_list, axis=1)
            test_lik = np.mean(test_lik, axis=1)
            test_lik = np.mean(test_lik)
            test_lik_list1.append(test_lik)
            path = saver.save(sess,FLAGS.logdir+"/model.ckpt",i)
            print("Model saved at ",path)
    
    # Save the summary data for analysis
    A_,DELTA_inv_ = sess.run([A_old,DELTA_inv])
    np.save("vtp_loss1.npy", vtp_list)
    #np.save("x_post_list1.npy",post_test_list)
    np.save("A1.npy",np.mean(A_, axis=0))
    np.save("delta_inv1.npy",np.mean(DELTA_inv_,axis=0))

if __name__ == "__main__":
    x = tf.placeholder(tf.float32, shape=(batch_size, inp_data_dim))
    z_sampled = tf.random_normal([batch_size, latent_dim])
    reverse_kl = tf.placeholder_with_default(1.0,())
    pearson = tf.placeholder_with_default(0.0,())
    eps = tf.random_normal([batch_size, eps_dim])

    n_samples=4
    U,V,DELTA_inv = generator(inp_data_dim,latent_dim,rank,n_samples=n_samples)
    U1 = tf.slice(U,[0,0,0],[1,-1,-1])
    V1 = tf.slice(V,[0,0,0],[1,-1,-1])
    M1 = tf.slice(M,[0,0,0],[1,-1,-1])
    A1 = tf.matmul(U1,V1)
    A1 = A1*M1
    DELTA_inv1 = tf.slice(DELTA_inv, [0,0],[1,-1])
    # Draw samples from posterior q(z2|x)
    print("Sampling from posterior...")
    eps = tf.random_normal(tf.stack([eps_nbasis, batch_size, eps_dim]))
    z, z_mean, z_var = encoder(x,eps,reuse=False)
    z_mean, z_var = tf.stop_gradient(z_mean), tf.stop_gradient(z_var)
    z_std = tf.sqrt(z_var + 1e-4)
    # z_sampled = R_std*R_sampled+R_mean
    z_norm = (z-1.0*z_mean)/z_std
    logz = get_zlogprob(z, "gauss")
    logr = -0.5 * tf.reduce_sum(z_norm*z_norm + tf.log(z_var) + latent_dim*np.log(2*np.pi), [1])
    logz = tf.reduce_mean(logz)
    logr = tf.reduce_mean(logr)

    # Evaluating p(x|z)
    means = tf.matmul(tf.ones([A1.get_shape().as_list()[0],z.get_shape().as_list()[0],latent_dim])*z,tf.transpose(A1, perm=[0,2,1]))
    prec = tf.square(DELTA_inv1)
    t = (x-means)
    t1 = t*tf.expand_dims(prec, axis=1)*t
    t1 = -0.5*tf.reduce_sum(t1, axis=2) # (n_samples, batch_size)
    t2 = 0.5*tf.expand_dims(tf.reduce_sum(tf.log(1e-3+prec), axis=1),axis=1) # (n_samples,1)
    t3 = -inp_data_dim*0.5*tf.log(2*math.pi)
    x_post_prob_log = t1+t2+t3
    x_post_prob_log = tf.reduce_mean(x_post_prob_log, axis=1)
    x_post_prob_log = tf.reduce_mean(x_post_prob_log)

    # Dual loss
    Td = data_network(z_norm)
    Ti = data_network(z_sampled, reuse=True)
    d_loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Td, labels=tf.ones_like(Td)))
    d_loss_i = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Ti, labels=tf.zeros_like(Ti)))
    dual_loss = d_loss_d+d_loss_i

    # dual loss for theta
    dual_loss_theta, label_acc_adv_theta, q_ratio = cal_theta_adv_loss(A1,DELTA_inv,inp_data_dim, latent_dim, rank)
    
    #Adversary Accuracy
    correct_labels_adv = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.greater(tf.sigmoid(Td),thresh_adv), tf.int32),1),tf.float32)) + tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.less_equal(tf.sigmoid(Td),thresh_adv), tf.int32),0),tf.float32))
    label_acc_adv = correct_labels_adv/(2.0*batch_size)

    # Primal loss
    t1 = -tf.reduce_mean(Ti)
    t2 = x_post_prob_log
    t5 = logz-logr
    t3 = q_ratio
    f_input = tf.squeeze(q_ratio)-Td
    f_input = tf.exp(f_input) 
    t4 = tf.reduce_mean(tf.square(f_input-1))
    KL_neg_r_q = t1
    ELBO = pearson*(t2-t4)+reverse_kl*(t1+t2+t3+t5)
    primal_loss = tf.reduce_mean(-ELBO)

    print("V in main:",V.get_shape().as_list())
    print("U in main:",U.get_shape().as_list())
    print("DELTA_inv in main:",DELTA_inv.get_shape().as_list())
    train(z, label_acc_adv_theta)
