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
latent_dim=60
nepoch = 1001
lr = 10**(-4)
batch_size = 150
ntrain = 300
test_batch_size = 34
ntest=34
inp_data_dim = 4732
inp_cov_dim = 3
eps_dim = 40
eps_nbasis=32
n_clf_epoch = 5000
thresh_adv = 0.5
rank = 30
num_hiv_classes = 2
num_tb_classes = 2
tb_coeff = 20
hiv_coeff = 20
""" tensorboard """
parser = argparse.ArgumentParser()
parser.add_argument('--logdir',type=str,default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),'tensorflow/mnist/logs/mnist_with_summaries'),help='Summaries log directory')
FLAGS, unparsed = parser.parse_known_args()
if tf.gfile.Exists(FLAGS.logdir):
    tf.gfile.DeleteRecursively(FLAGS.logdir)
tf.gfile.MakeDirs(FLAGS.logdir)

def load_dataset():
    raw_data = np.load("/opt/data/saket/gene_data/data/mod_gauss_data.npy")
    cov = np.load("/opt/data/saket/gene_data/data/reg.npy")
    labels = np.load("/opt/data/saket/gene_data/data/hiv.npy")
    labels_tb = np.load("/opt/data/saket/gene_data/data/tb.npy")
    labels = np.squeeze(labels)
    labels_tb = np.squeeze(labels_tb)
    global inp_data_dim
    global inp_cov_dim
    inp_data_dim = raw_data.shape[1]
    inp_cov_dim = cov.shape[1]
    m = np.mean(raw_data, axis=0)
    raw_data = (raw_data-m)/5.0
    #cov = (np.log10(cov+0.1))/5.0
    return raw_data[0:ntrain],cov[0:ntrain],labels[0:ntrain],labels_tb[0:ntrain],raw_data[ntrain:],cov[ntrain:],labels[ntrain:],labels_tb[ntrain:]

X_train, C_train, L_train, Ltb_train, X_test, C_test, L_test, Ltb_test1 = load_dataset()
print("inp_data_dim:",inp_data_dim)
print("inp_cov_dim:",inp_cov_dim)
X_test = X_test.astype(np.float32)
C_test = C_test.astype(np.float32)
n1 = (Ltb_test1==1).nonzero()[0].shape[0]
n0 = (Ltb_test1==0).nonzero()[0].shape[0]
n2 = (Ltb_test1==2).nonzero()[0].shape[0]
print("Input: For ","Tb"," #1:",n1," #0:",n0, " #2:",n2)
Lac_test = np.copy(Ltb_test1)
Ltb_test = np.copy(Ltb_test1)
Lla_test = np.copy(Ltb_test1)
i2 = (Ltb_test==2).nonzero()[0]
i1 = (Ltb_test==1).nonzero()[0]
i0 = (Ltb_test==0).nonzero()[0]
Ltb_test[i2] = 1
i2 = (Lac_test==2).nonzero()[0]
i1 = (Lac_test==1).nonzero()[0]
i0 = (Lac_test==0).nonzero()[0]
Lac_test[i1] = 0
Lac_test[i2] = 1
i2 = (Lla_test==2).nonzero()[0]
i1 = (Lla_test==1).nonzero()[0]
i0 = (Lla_test==0).nonzero()[0]
Lla_test[i2] = 0

n1 = (Ltb_test==1).nonzero()[0].shape[0]
n0 = (Ltb_test==0).nonzero()[0].shape[0]
print("Input: For ","Tb"," #1:",n1," #0:",n0)

n1 = (Lac_test==1).nonzero()[0].shape[0]
n0 = (Lac_test==0).nonzero()[0].shape[0]
print("Input: For ","AC"," #1:",n1," #0:",n0)

n1 = (Lla_test==1).nonzero()[0].shape[0]
n0 = (Lla_test==0).nonzero()[0].shape[0]
print("Input: For ","La"," #1:",n1," #0:",n0)
l1_regulariser = tf.contrib.layers.l1_regularizer(0.05)
lb = skp.LabelBinarizer()
lb.fit([0,1])

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

def train(z, closs, label_acc_adv_theta):
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith("data_net")]
    c_vars = [var for var in t_vars if var.name.startswith("Classifier") or var.name.startswith("Classifier_tb") or var.name.startswith("Classifier_active_tb") or var.name.startswith("Classifier_latent_tb")]
    tr_vars = [var for var in t_vars if (var.name.startswith("U") or var.name.startswith("V") or var.name.startswith("B") or var.name.startswith("del") or var.name.startswith("M") or var.name.startswith("Mtb") or var.name.startswith("Mactive") or var.name.startswith("Mlatent") or var.name.startswith("Mhiv") or var.name.startswith("Wtb") or var.name.startswith("Wactive") or var.name.startswith("Wlatent") or var.name.startswith("Whiv"))]
    tp_var = [var for var in t_vars if var not in d_vars+c_vars+tr_vars]
    #print("tp_var:",tp_var)
    #print("tr_var:",tr_vars)
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
    U_test,V_test,B_test,D_test,M_test,Mtb_test, Mactive_test,Mlatent_test,Mhiv_test,W_tb_test,W_active_test,W_latent_test,W_hiv_test = generator(inp_data_dim,inp_cov_dim,latent_dim,rank,n_samples=500, reuse=True)
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
    prob_test1, closs_test1 = classifier(z_test,labels,W_hiv_test,Mhiv_test,reuse=True)
    print("prob_Test1:",prob_test1.shape)
    correct_label_pred_test = tf.equal(tf.argmax(prob_test1,1),labels)
    label_acc_test1 = tf.reduce_mean(tf.cast(correct_label_pred_test, tf.float32))

    prob_test2, closs_test2 = classifier_tb(z_test,labels_tb,W_tb_test,Mtb_test,reuse=True)
    correct_label_pred_test = tf.equal(tf.argmax(prob_test2,1),labels_tb)
    label_acc_test2 = tf.reduce_mean(tf.cast(correct_label_pred_test, tf.float32))

    prob_test3, closs_test3 = classifier_active_tb(z_test,labels_active,W_active_test,Mactive_test,reuse=True)
    correct_label_pred_test = tf.equal(tf.argmax(prob_test3,1),labels_active)
    label_acc_test3 = tf.reduce_mean(tf.cast(correct_label_pred_test, tf.float32))

    prob_test4, closs_test4 = classifier_latent_tb(z_test,labels_latent,W_latent_test,Mlatent_test,reuse=True)
    correct_label_pred_test = tf.equal(tf.argmax(prob_test4,1),labels_latent)
    label_acc_test4 = tf.reduce_mean(tf.cast(correct_label_pred_test, tf.float32))

    closs_test = closs_test1+closs_test2+closs_test3+closs_test4
    label_acc_test = (label_acc_test1+label_acc_test2+label_acc_test3+label_acc_test4)/4.0
    
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
    auc=[]
    for i in range(nepoch):
        for j in range(ntrain//batch_size):
            xmb = X_train[j*batch_size:(j+1)*batch_size]
            cmb = C_train[j*batch_size:(j+1)*batch_size]
            ltbmb = np.copy(Ltb_train[j*batch_size:(j+1)*batch_size])
            i2 = (ltbmb == 2).nonzero()[0]
            ltbmb[i2] = 1
            lacmb = np.copy(Ltb_train[j*batch_size:(j+1)*batch_size])
            i1 = (lacmb == 1).nonzero()[0]
            lacmb[i1] = 0
            i2 = (lacmb == 2).nonzero()[0]
            lacmb[i2] = 1
            llamb = np.copy(Ltb_train[j*batch_size:(j+1)*batch_size])
            i2 = (llamb == 2).nonzero()[0]
            llamb[i2] = 0
            # sess.run(sample_from_r, feed_dict={x:xmb, c:cmb})
            for gen in range(1):
                sess.run(train_op, feed_dict={x:xmb,c:cmb,labels:L_train[j*batch_size:(j+1)*batch_size],labels_tb:ltbmb, labels_active:lacmb, labels_latent:llamb})
            for gen in range(1):
                sess.run(adversary_theta_train_op, feed_dict={x:xmb,c:cmb,labels:L_train[j*batch_size:(j+1)*batch_size],labels_tb:ltbmb, labels_active:lacmb, labels_latent:llamb})
            vtp_loss,closs_,closstb_,clossac_,clossla_,label_hiv,label_tb,label_active,label_latent = sess.run([primal_loss, closs1, closs2, closs3, closs4, label_acc1,label_acc2,label_acc3,label_acc4], feed_dict={x:xmb,c:cmb,labels:L_train[j*batch_size:(j+1)*batch_size],labels_tb:ltbmb, labels_active:lacmb, labels_latent:llamb})
            clf_loss_list.append((closs_, closstb_,clossac_,clossla_))
            vtp_list.append(vtp_loss)
        if i%100 == 0:
            Td_,KL_neg_r_q_,x_post_prob_log_,logz_,logr_,d_loss_d_,d_loss_i_,label_acc_adv_,label_acc_adv_theta_,dual_loss_theta_ = sess.run([Td,KL_neg_r_q,x_post_prob_log,logz,logr,d_loss_d,d_loss_i,label_acc_adv, label_acc_adv_theta, dual_loss_theta], feed_dict={x:xmb,c:cmb})
            print("epoch:",i," vtp:",vtp_list[-1], " x_post:",x_post_prob_log_," closs:",closs_," label_acc_hiv:",label_hiv," label_acc_tb:",label_tb, " label_acc_active:",label_active, " label_acc_latent:",label_latent)
        if i%500 == 0:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary = sess.run(merged,feed_dict={x:xmb,c:cmb})
            train_writer.add_run_metadata(run_metadata, 'step %i'%(i))
            train_writer.add_summary(summary, i)

        if i%100 == 0:
            test_lik_list = []
            test_prob_hiv = []
            test_prob_tb = []
            test_prob_ac = []
            test_prob_la = []
            test_acc_tb = []
            for i in range(100):
                test_lik = sess.run(x_post_prob_log_test)
                test_lik_list.append(test_lik)
                lt_hiv,lt_tb,lt_ac,lt_la, la, la_hiv, la_tb = sess.run([prob_test1, prob_test2,prob_test3,prob_test4, label_acc_test, label_acc_test1, label_acc_test2], feed_dict={labels:L_test,labels_tb:Ltb_test, labels_active:Lac_test,labels_latent:Lla_test})
                test_prob_hiv.append(lt_hiv)
                test_prob_tb.append(lt_tb)
                test_prob_ac.append(lt_ac)
                test_prob_la.append(lt_la)
                test_acc_tb.append(la_tb)
            avg_test_prob_hiv = np.mean(test_prob_hiv,axis=0)
            avg_test_prob_tb = np.mean(test_prob_tb,axis=0)
            #print("avg_test_prob_tb:",avg_test_prob_tb)
            #print("Ltb_test:",Ltb_test)
            avg_acc_tb_tmp = np.mean(test_acc_tb)
            print("tmp tb acc:",avg_acc_tb_tmp)
            avg_test_prob_ac = np.mean(test_prob_ac,axis=0)
            avg_test_prob_la = np.mean(test_prob_la,axis=0)
            avg_test_acc_hiv = np.mean((np.argmax(avg_test_prob_hiv,axis=1)==L_test))
            avg_test_acc_tb = np.mean((np.argmax(avg_test_prob_tb,axis=1)==Ltb_test))
            avg_test_acc_ac = np.mean((np.argmax(avg_test_prob_ac,axis=1)==Lac_test))
            avg_test_acc_la = np.mean((np.argmax(avg_test_prob_la,axis=1)==Lla_test))
            print("Average Test Set Accuracy HIV:",avg_test_acc_hiv)
            print("Average Test Set Accuracy TB:",avg_test_acc_tb)
            print("Average Test Set Accuracy Active TB:",avg_test_acc_ac)
            print("Average Test Set Accuracy Latent TB:",avg_test_acc_la)
            countN1(avg_test_prob_tb,Ltb_test,"Tb")
            countN1(avg_test_prob_ac,Lac_test,"Active Tb")
            countN1(avg_test_prob_la,Lla_test,"Latent Tb")
            tmp = np.zeros((L_test.shape[0],2))
            #print("L_test dtype:",L_test.dtype)
            t = L_test.astype(np.int32)
            tmp[np.arange(L_test.shape[0]),t] = 1
            print("tmp shape:",tmp.shape)
            auc_hiv = skm.roc_auc_score(tmp,avg_test_prob_hiv)
            print("AUC HIV vs others:",auc_hiv)
            tmp = np.zeros((L_test.shape[0],2))
            t = Ltb_test.astype(np.int32)
            tmp[np.arange(L_test.shape[0]),t] = 1
            auc_tb = skm.roc_auc_score(tmp,avg_test_prob_tb)
            print("AUC TB vs others:",auc_tb)
            tmp = np.zeros((L_test.shape[0],2))
            t = Lac_test.astype(np.int32)
            tmp[np.arange(L_test.shape[0]),t] = 1
            auc_ac = skm.roc_auc_score(tmp,avg_test_prob_ac)
            print("AUC Active TB vs others:",auc_ac)
            tmp = np.zeros((L_test.shape[0],2))
            t = Lla_test.astype(np.int32)
            tmp[np.arange(L_test.shape[0]),t] = 1
            auc_la = skm.roc_auc_score(tmp,avg_test_prob_la)
            print("AUC Latent TB vs others:",auc_la)
            test_acc_list.append((avg_test_acc_hiv,avg_test_acc_tb))
            test_lik = np.stack(test_lik_list, axis=1)
            test_lik = np.mean(test_lik, axis=1)
            test_lik = np.mean(test_lik)
            test_lik_list1.append(test_lik)
            path = saver.save(sess,FLAGS.logdir+"/model.ckpt",i)
            print("Model saved at ",path)
            auc.append([auc_hiv,auc_tb,auc_ac,auc_la])
    
    # Save the summary data for analysis
    A_,B_,DELTA_inv_,M_test_,Mtb_test_, Mactive_test_,Mlatent_test_,Mhiv_test_ = sess.run([A,B,DELTA_inv, M_test,Mtb_test, Mactive_test,Mlatent_test,Mhiv_test])
    M_test_ = M_test_[0]
    Mtb_test_ = Mtb_test_[0]
    Mactive_test_ = Mactive_test_[0]
    Mlatent_test_ = Mlatent_test_[0]
    Mhiv_test_ = Mhiv_test_[0]
    np.save("vtp_loss1.npy", vtp_list)
    #np.save("x_post_list1.npy",post_test_list)
    np.save("A1.npy",np.mean(A_, axis=0))
    np.save("B1.npy",np.mean(B_,axis=0))
    np.save("delta_inv1.npy",np.mean(DELTA_inv_,axis=0))
    np.save("clf_loss_list1.npy",clf_loss_list)
    np.save("test_lik.npy",test_lik_list1)
    np.save("test_acc.npy",test_acc_list)
    np.save("M1.npy",M_test_)
    np.save("auc.npy",auc)
    np.save("Mtb.npy",Mtb_test_)
    np.save("Mactive.npy",Mactive_test_)
    np.save("Mlatent.npy",Mlatent_test_)
    np.save("Mhiv.npy",Mhiv_test_)

    # Test Set
    z_list = []       
    test_lik_list = []
    test_prob_hiv = []
    test_prob_tb = []
    test_prob_ac = []
    test_prob_la = []
    for i in range(100):
        lt_hiv,lt_tb,lt_ac,lt_la, la, la_hiv, la_tb = sess.run([prob_test1, prob_test2,prob_test3,prob_test4, label_acc_test, label_acc_test1, label_acc_test2], feed_dict={labels:L_test,labels_tb:Ltb_test, labels_active:Lac_test,labels_latent:Lla_test})
        test_prob_hiv.append(lt_hiv)
        test_prob_tb.append([lt_tb])
        test_prob_ac.append([lt_ac])
        test_prob_la.append([lt_la])
    avg_test_prob_hiv = np.mean(test_prob_hiv,axis=0)
    avg_test_prob_tb = np.mean(test_prob_tb,axis=0)
    avg_test_prob_ac = np.mean(test_prob_ac,axis=0)
    avg_test_prob_la = np.mean(test_prob_la,axis=0)
    avg_test_acc_hiv = np.mean((np.argmax(avg_test_prob_hiv,axis=1)==L_test))
    avg_test_acc_tb = np.mean((np.argmax(avg_test_prob_tb,axis=1)==Ltb_test))
    avg_test_acc_ac = np.mean((np.argmax(avg_test_prob_ac,axis=1)==Lac_test))
    avg_test_acc_la = np.mean((np.argmax(avg_test_prob_la,axis=1)==Lla_test))
    print("Average Test Set Accuracy HIV:",avg_test_acc_hiv)
    print("Average Test Set Accuracy TB:",avg_test_acc_tb)
    print("Average Test Set Accuracy Active TB:",avg_test_acc_ac)
    print("Average Test Set Accuracy Latent TB:",avg_test_acc_la)

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
    labels_tb = tf.placeholder(tf.int64, shape=(None))
    labels_active = tf.placeholder(tf.int64, shape=(None))
    labels_latent = tf.placeholder(tf.int64, shape=(None))
    eps = tf.random_normal([batch_size, eps_dim])

    n_samples=4
    U,V,B,DELTA_inv,M,M_tb,M_active,M_latent,M_hiv,W_tb,W_active,W_latent,W_hiv = generator(inp_data_dim,inp_cov_dim,latent_dim,rank,n_samples=n_samples)
    U1 = tf.slice(U,[0,0,0],[1,-1,-1])
    V1 = tf.slice(V,[0,0,0],[1,-1,-1])
    M1 = tf.slice(M,[0,0,0],[1,-1,-1])
    A1 = tf.matmul(U1,V1)
    A1 = A1*M1
    B1 = tf.slice(B,[0,0,0],[1,-1,-1])
    DELTA_inv1 = tf.slice(DELTA_inv, [0,0],[1,-1])
    M_tb = tf.slice(M_tb,[0,0],[1,-1])
    M_active = tf.slice(M_active,[0,0],[1,-1])
    M_latent = tf.slice(M_latent,[0,0],[1,-1])
    M_hiv = tf.slice(M_hiv,[0,0],[1,-1])
    W_tb = tf.slice(W_tb,[0,0],[1,-1])
    W_active = tf.slice(W_active,[0,0],[1,-1])
    W_latent = tf.slice(W_latent,[0,0],[1,-1])
    W_hiv = tf.slice(W_hiv,[0,0],[1,-1])
    #M_tb = tf.Print(M_tb,[M_tb],message="M_tb")
    #print("W_tb_shape:",W_tb.get_shape().as_list())

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
    logits, closs1 = classifier(z,labels,W_hiv,M_hiv,reuse=False)
    correct_label_pred = tf.equal(tf.argmax(logits,1),labels)
    label_acc1 = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))

    logits, closs2 = classifier_tb(z,labels_tb,W_tb,M_tb,reuse=False)
    correct_label_pred = tf.equal(tf.argmax(logits,1),labels_tb)
    label_acc2 = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))

    logits, closs3 = classifier_active_tb(z,labels_active,W_active,M_active,reuse=False)
    correct_label_pred = tf.equal(tf.argmax(logits,1),labels_active)
    #correct_label_pred = tf.Print(correct_label_pred,[correct_label_pred],message="Avtive pred train")
    label_acc3 = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))

    logits, closs4 = classifier_latent_tb(z,labels_latent,W_latent,M_latent,reuse=False)
    correct_label_pred = tf.equal(tf.argmax(logits,1),labels_latent)
    label_acc4 = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))

    closs = hiv_coeff*closs1+tb_coeff*(closs2+closs3+closs4)
    label_acc = (label_acc1+label_acc2+label_acc3+label_acc4)/4.0

    # Dual loss
    Td = data_network(z_norm)
    Ti = data_network(z_sampled, reuse=True)
    d_loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Td, labels=tf.ones_like(Td)))
    d_loss_i = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Ti, labels=tf.zeros_like(Ti)))
    dual_loss = d_loss_d+d_loss_i

    # dual loss for theta
    dual_loss_theta, label_acc_adv_theta, q_ratio = cal_theta_adv_loss(A1,B,DELTA_inv,inp_data_dim, inp_cov_dim, latent_dim, rank)
    
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
