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
latent_dim=30
nepoch = 2401
lr = 10**(-4)
batch_size = 440
ntrain = 1000
test_batch_size = 207
ntest=207
inp_data_dim = 5000
inp_cov_dim = 7
eps_dim = 40
eps_nbasis=32
n_clf_epoch = 5000
thresh_adv = 0.5
rank = 15
num_hiv_classes = 2
num_tb_classes = 2
tb_coeff = 60
""" tensorboard """
parser = argparse.ArgumentParser()
parser.add_argument('--logdir',type=str,default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),'tensorflow/mnist/logs/mnist_with_summaries'),help='Summaries log directory')
FLAGS, unparsed = parser.parse_known_args()
if tf.gfile.Exists(FLAGS.logdir):
    tf.gfile.DeleteRecursively(FLAGS.logdir)
tf.gfile.MakeDirs(FLAGS.logdir)

def load_dataset():
    raw_data = np.load("data/data_micro/mod_feat_micro.npy")
    cov = np.load("data/data_micro/micro_cov.npy")
    labels_cnp = np.load("data/data_micro/micro_label_cnp.npy")
    labels_diabetes = np.load("data/data_micro/micro_label_diabetes.npy")
    labels_gluten = np.load("data/data_micro/micro_label_gluten.npy")
    labels_ibd = np.load("data/data_micro/micro_label_ibd.npy")
    labels_lactose = np.load("data/data_micro/micro_label_lactose.npy")
    labels_quino = np.load("data/data_micro/micro_label_quinoline.npy")
    labels_cnp = np.squeeze(labels_cnp)
    labels_diabetes = np.squeeze(labels_diabetes)
    labels_gluten = np.squeeze(labels_gluten)
    labels_ibd = np.squeeze(labels_ibd)
    labels_lactose = np.squeeze(labels_lactose)
    labels_quino = np.squeeze(labels_quino)
    global inp_data_dim
    global inp_cov_dim
    inp_data_dim = raw_data.shape[1]
    inp_cov_dim = cov.shape[1]
    m = np.mean(raw_data, axis=0)
    raw_data = (raw_data-m)/4.0
    cov = (np.log10(cov+0.1))
    return raw_data[0:ntrain],cov[0:ntrain],labels_cnp[0:ntrain],labels_diabetes[0:ntrain],labels_gluten[0:ntrain],labels_ibd[0:ntrain],labels_lactose[0:ntrain],labels_quino[0:ntrain],raw_data[ntrain:],cov[ntrain:],labels_cnp[ntrain:],labels_diabetes[ntrain:],labels_gluten[ntrain:],labels_ibd[ntrain:],labels_lactose[ntrain:],labels_quino[ntrain:]

X_train, C_train, Lcnp_train, Ldiabetes_train, Lgluten_train, Libd_train, Llactose_train, Lquino_train, X_test, C_test, Lcnp_test, Ldiabetes_test, Lgluten_test, Libd_test, Llactose_test, Lquino_test = load_dataset()
def next_minibatch():
    cnp0 = (Lcnp_train==0).nonzero()[0]
    cnp1 = (Lcnp_train==1).nonzero()[0]
    db0 = (Ldiabetes_train==0).nonzero()[0]
    db1 = (Ldiabetes_train==1).nonzero()[0]
    gl0 = (Lgluten_train==0).nonzero()[0]
    gl1 = (Lgluten_train==1).nonzero()[0]
    ibd0 = (Libd_train==0).nonzero()[0]
    ibd1 = (Libd_train==1).nonzero()[0]
    quino0 = (Lquino_train==0).nonzero()[0]
    quino1 = (Lquino_train==1).nonzero()[0]
#    indices = np.concatenate([np.random.choice(cnp0,80),np.random.choice(cnp1,80),np.random.choice(db0,20),np.random.choice(db1,20),np.random.choice(ibd0,90),np.random.choice(ibd1,30),np.random.choice(quino0,30),np.random.choice(quino1,30)])
    indices = np.concatenate([np.random.choice(gl0,200),np.random.choice(gl1,200),np.random.choice(ibd1,40)])
    xmb = X_train[indices,:]
    cmb = C_train[indices,:]
    cnp_mb = Lcnp_train[indices]
    db_mb = Ldiabetes_train[indices]
    gl_mb = Lgluten_train[indices]
    ibd_mb = Libd_train[indices]
    lac_mb = Llactose_train[indices]
    quin_mb = Lquino_train[indices]
    return xmb,cmb,cnp_mb,db_mb,gl_mb,ibd_mb,lac_mb,quin_mb
print("inp_data_dim:",inp_data_dim)
print("inp_cov_dim:",inp_cov_dim)
X_test = X_test.astype(np.float32)
C_test = C_test.astype(np.float32)

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
    c_vars = [var for var in t_vars if var.name.startswith("Classifier_cnp") or var.name.startswith("Classifier_diabetes") or var.name.startswith("Classifier_gluten") or var.name.startswith("Classifier_ibd") or var.name.startswith("Classifier_lactose") or var.name.startswith("Classifier_quino")]
    tr_vars = [var for var in t_vars if (var.name.startswith("U") or var.name.startswith("V") or var.name.startswith("B") or var.name.startswith("del") or var.name.startswith("M") or var.name.startswith("Mcnp") or var.name.startswith("Mdiabetes") or var.name.startswith("Mgluten") or var.name.startswith("Mibd") or var.name.startswith("Mlactose") or var.name.startswith("Mquino") or var.name.startswith("Wcnp") or var.name.startswith("Wdiabetes") or var.name.startswith("Wgluten") or var.name.startswith("Wibd") or var.name.startswith("Wlactose") or var.name.startswith("Wquino"))]
    tp_var = [var for var in t_vars if var not in d_vars+c_vars+tr_vars]
    #print("tp_var:",tp_var)
    #print("tr_var:",tr_vars)
    assert(len(tp_var)+len(d_vars)+len(c_vars)+len(tr_vars) == len(t_vars))

    r_loss = tf.losses.get_regularization_loss()
    r_loss_clf = tf.losses.get_regularization_loss(scope="Classifier")
    r_loss -= r_loss_clf
    primal_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3, use_locking=True, beta1=0.5)
    dual_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, use_locking=True, beta1=0.5)
    dual_optimizer_theta = tf.train.AdamOptimizer(learning_rate=1e-4, use_locking=True, beta1=0.5)
    #classifier_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3, use_locking=True)

    #primal_train_op = primal_optimizer.minimize(primal_loss+r_loss+closs+r_loss_clf, var_list=tp_var+c_vars)
    #adversary_train_op = dual_optimizer.minimize(dual_loss+r_loss, var_list=d_vars)
    #adversary_theta_train_op = dual_optimizer_theta.minimize(dual_loss_theta+r_loss, var_list=tr_vars)
    #clf_train_op = classifier_optimizer.minimize(closs+r_loss_clf, var_list=c_vars)

    # Clipping Gradients
    primal_grad = primal_optimizer.compute_gradients(primal_loss+r_loss+closs+r_loss_clf, var_list=tp_var+c_vars)
    capped_g_grad = []
    for grad,var in primal_grad:
        if grad is not None:
            capped_g_grad.append((tf.clip_by_value(grad,-0.1,0.1),var))
    primal_train_op = primal_optimizer.apply_gradients(capped_g_grad)
 
    #adversary_train_op = dual_optimizer.minimize(dual_loss+r_loss, var_list=d_vars)
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
    z_test, _, _ = encoder(X_test,C_test,eps,reuse=True)
    U_test,V_test,B_test,D_test,M_test,Mgluten_test,Mibd_test,W_gluten_test,W_ibd_test = generator(inp_data_dim,inp_cov_dim,latent_dim,rank,keep_prob,n_samples=500, reuse=True)
    A_old = tf.matmul(U_test,V_test)
    A = A_old*M_test
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

    #prob_test2, closs_test2 = classifier_cnp(z_test,labels_cnp,W_cnp_test,Mcnp_test,reuse=True)
    #correct_label_pred_test = tf.equal(tf.argmax(prob_test2,1),labels_cnp)
    #label_acc_test2 = tf.reduce_mean(tf.cast(correct_label_pred_test, tf.float32))

    #prob_test3, closs_test3 = classifier_diabetes(z_test,labels_diabetes,W_diabetes_test,Mdiabetes_test,reuse=True)
    #correct_label_pred_test = tf.equal(tf.argmax(prob_test3,1),labels_diabetes)
    #label_acc_test3 = tf.reduce_mean(tf.cast(correct_label_pred_test, tf.float32))

    prob_test4, closs_test4 = classifier_gluten(z_test,labels_gluten,W_gluten_test,Mgluten_test,reuse=True)
    correct_label_pred_test = tf.equal(tf.argmax(prob_test4,1),labels_gluten)
    label_acc_test4 = tf.reduce_mean(tf.cast(correct_label_pred_test, tf.float32))

    prob_test5, closs_test5 = classifier_ibd(z_test,labels_ibd,W_ibd_test,Mibd_test,reuse=True)
    correct_label_pred_test = tf.equal(tf.argmax(prob_test4,1),labels_ibd)
    label_acc_test5 = tf.reduce_mean(tf.cast(correct_label_pred_test, tf.float32))

    #prob_test6, closs_test6 = classifier_lactose(z_test,labels_lactose,W_lactose_test,Mlactose_test,reuse=True)
    #correct_label_pred_test = tf.equal(tf.argmax(prob_test4,1),labels_lactose)
    #label_acc_test6 = tf.reduce_mean(tf.cast(correct_label_pred_test, tf.float32))

    #prob_test7, closs_test7 = classifier_quino(z_test,labels_quino,W_quino_test,Mquino_test,reuse=True)
    #correct_label_pred_test = tf.equal(tf.argmax(prob_test4,1),labels_quino)
    #label_acc_test7 = tf.reduce_mean(tf.cast(correct_label_pred_test, tf.float32))

    closs_test =closs_test4+closs_test5
    label_acc_test = (label_acc_test4+label_acc_test5)/2.0

    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.logdir, graph = tf.get_default_graph())
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    #saver.restore(sess,"model/model.ckpt-99")
    si_list = []
    vtp_list = []
    clf_loss_list = []
    post_test_list = []
    dt_list = []
    test_lik_list1 = []
    test_acc_list = []
    auc=[]
    val = 1
    for i in range(nepoch):
        if i > 50:
            val = 0
        for j in range(ntrain//batch_size):
            xmb = X_train[j*batch_size:(j+1)*batch_size]
            cmb = C_train[j*batch_size:(j+1)*batch_size]
            # sess.run(sample_from_r, feed_dict={x:xmb, c:cmb})
            xmb,cmb,cnp_mb,db_mb,gl_mb,ibd_mb,lac_mb,quin_mb = next_minibatch()
            for gen in range(1):
                sess.run(train_op, feed_dict={x:xmb,c:cmb,labels_cnp:cnp_mb,labels_diabetes:db_mb,labels_gluten:gl_mb,labels_ibd:ibd_mb,labels_lactose:lac_mb,labels_quino:quin_mb,keep_prob:0.2})
                #sess.run(train_op, feed_dict={x:xmb,c:cmb,labels_cnp:Lcnp_train[j*batch_size:(j+1)*batch_size],labels_diabetes:Ldiabetes_train[j*batch_size:(j+1)*batch_size],labels_gluten:Lgluten_train[j*batch_size:(j+1)*batch_size],labels_ibd:Libd_train[j*batch_size:(j+1)*batch_size],labels_lactose:Llactose_train[j*batch_size:(j+1)*batch_size],labels_quino:Lquino_train[j*batch_size:(j+1)*batch_size],keep_prob:0.3, reverse_kl:0, pearson:1})
            for gen in range(1):
                sess.run(adversary_theta_train_op, feed_dict={x:xmb,c:cmb,labels_cnp:cnp_mb,labels_diabetes:db_mb,labels_gluten:gl_mb,labels_ibd:ibd_mb,labels_lactose:lac_mb,labels_quino:quin_mb,keep_prob:0.2})
                #sess.run(adversary_theta_train_op, feed_dict={x:xmb,c:cmb,labels_cnp:Lcnp_train[j*batch_size:(j+1)*batch_size],labels_diabetes:Ldiabetes_train[j*batch_size:(j+1)*batch_size],labels_gluten:Lgluten_train[j*batch_size:(j+1)*batch_size],labels_ibd:Libd_train[j*batch_size:(j+1)*batch_size],labels_lactose:Llactose_train[j*batch_size:(j+1)*batch_size],labels_quino:Lquino_train[j*batch_size:(j+1)*batch_size],keep_prob:0.3, reverse_kl:0, pearson:1})

            vtp_loss,closs_,clossgluten_,clossibd_,label_gluten,label_ibd = sess.run([primal_loss, closs, closs4, closs5,label_acc4,label_acc5,], feed_dict={x:xmb,c:cmb,labels_cnp:cnp_mb,labels_diabetes:db_mb,labels_gluten:gl_mb,labels_ibd:ibd_mb,labels_lactose:lac_mb,labels_quino:quin_mb,keep_prob:0.2})
            clf_loss_list.append((closs_,clossgluten_,clossibd_))
            vtp_list.append(vtp_loss)
        if i%100 == 0:
            Td_,KL_neg_r_q_,x_post_prob_log_,logz_,logr_,d_loss_d_,d_loss_i_,label_acc_adv_,label_acc_adv_theta_,dual_loss_theta_,f_input_ = sess.run([Td,KL_neg_r_q,x_post_prob_log,logz,logr,d_loss_d,d_loss_i,label_acc_adv, label_acc_adv_theta, dual_loss_theta,f_input], feed_dict={x:xmb,c:cmb})
            print("epoch:",i," vtp:",vtp_list[-1], " x_post:",x_post_prob_log_," closs:",closs_, " label_acc_gluten:",label_gluten, " label_acc_ibd:", " logr:",logr_," logz:",logz_)
        if i%500 == 0:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary = sess.run(merged,feed_dict={x:xmb,c:cmb})
            train_writer.add_run_metadata(run_metadata, 'step %i'%(i))
            train_writer.add_summary(summary, i)

        if i%100 == 0:
            test_lik_list = []
            #test_prob_cnp = []
            #test_prob_db = []
            test_prob_gl = []
            test_prob_ibd = []
            #test_prob_lac = []
            #test_prob_quino = []
            test_acc_tb = []
            for i in range(100):
                test_lik = sess.run(x_post_prob_log_test)
                test_lik_list.append(test_lik)
                lt_gl,lt_ibd, la  = sess.run([prob_test4,prob_test5, label_acc_test], feed_dict={labels_cnp:Lcnp_test,labels_diabetes:Ldiabetes_test,labels_gluten:Lgluten_test,labels_ibd:Libd_test,labels_lactose:Llactose_test,labels_quino:Lquino_test})

                test_prob_gl.append(lt_gl)
                test_prob_ibd.append(lt_ibd)
            #print("avg_test_prob_tb:",avg_test_prob_tb)
            #print("Ltb_test:",Ltb_test)
            avg_test_prob_gl = np.mean(test_prob_gl,axis=0)
            avg_test_prob_ibd = np.mean(test_prob_ibd,axis=0)
            avg_test_acc_gl = np.mean((np.argmax(avg_test_prob_gl,axis=1)==Lgluten_test))
            avg_test_acc_ibd = np.mean((np.argmax(avg_test_prob_ibd,axis=1)==Libd_test))
            print("Average Test Set Accuracy Gluten:",avg_test_acc_gl)
            print("Average Test Set Accuracy ibd:",avg_test_acc_ibd)
            countN1(avg_test_prob_gl,Lgluten_test,"Gluten")
            countN1(avg_test_prob_ibd,Libd_test,"IBD")
            tmp = np.zeros((Lgluten_test.shape[0],2))
            t = Lgluten_test.astype(np.int32)
            tmp[np.arange(Lgluten_test.shape[0]),t] = 1
            auc_gl = skm.roc_auc_score(tmp,avg_test_prob_gl)
            print("AUC Gluten:",auc_gl)
            tmp = np.zeros((Libd_test.shape[0],2))
            t = Libd_test.astype(np.int32)
            tmp[np.arange(Libd_test.shape[0]),t] = 1
            auc_ibd = skm.roc_auc_score(tmp,avg_test_prob_ibd)
            print("AUC IBD:",auc_ibd)
            test_lik = np.stack(test_lik_list, axis=1)
            test_lik = np.mean(test_lik, axis=1)
            test_lik = np.mean(test_lik)
            test_lik_list1.append(test_lik)
            path = saver.save(sess,FLAGS.logdir+"/model.ckpt",i)
            print("Model saved at ",path)
            auc.append([auc_gl,auc_ibd])

    # Save the summary data for analysis
    A_,B_,DELTA_inv_,M_test_,Mgluten_test_,Mibd_test_ = sess.run([A_old,B,DELTA_inv, M_test,Mgluten_test,Mibd_test])
    M_test_ = M_test_[0]
    Mgluten_test_ = Mgluten_test_[0]
    Mibd_test_ = Mibd_test_[0]
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
    np.save("Mgluten.npy",Mgluten_test_)
    np.save("Mibd.npy",Mibd_test_)
    ## Test Set
    #z_list = []       
    #test_lik_list = []
    #test_prob_tb = []
    #test_prob_ac = []
    #test_prob_la = []
    #for i in range(100):
    #    lt_tb,lt_ac,lt_la, la, la_tb = sess.run([prob_test2,prob_test3,prob_test4, label_acc_test,label_acc_test2], feed_dict={labels_tb:Ltb_test, labels_active:Lac_test,labels_latent:Lla_test})
    #    test_prob_tb.append(lt_tb)
    #    test_prob_ac.append(lt_ac)
    #    test_prob_la.append(lt_la)
    #avg_test_prob_tb = np.mean(test_prob_tb,axis=0)
    #avg_test_prob_ac = np.mean(test_prob_ac,axis=0)
    #avg_test_prob_la = np.mean(test_prob_la,axis=0)
    #avg_test_acc_tb = np.mean((np.argmax(avg_test_prob_tb,axis=1)==Ltb_test))
    #avg_test_acc_ac = np.mean((np.argmax(avg_test_prob_ac,axis=1)==Lac_test))
    #avg_test_acc_la = np.mean((np.argmax(avg_test_prob_la,axis=1)==Lla_test))
    #print("Average Test Set Accuracy TB:",avg_test_acc_tb)
    #print("Average Test Set Accuracy Active TB:",avg_test_acc_ac)
    #print("Average Test Set Accuracy Latent TB:",avg_test_acc_la)

    #for i in range(20):
    #    z_ = sess.run(z_test)
    #    #print("z_:",z_)
    #    z_list.append(z_)
    #np.save("z_test_list.npy",z_list)

if __name__ == "__main__":
    x = tf.placeholder(tf.float32, shape=(batch_size, inp_data_dim))
    c = tf.placeholder(tf.float32, shape=(batch_size, inp_cov_dim))
    keep_prob = tf.placeholder_with_default(1.0,())
    reverse_kl = tf.placeholder_with_default(1.0,())
    pearson = tf.placeholder_with_default(0.0,())
    z_sampled = tf.random_normal([batch_size, latent_dim])
    labels_cnp = tf.placeholder(tf.int64, shape=(None))
    labels_diabetes = tf.placeholder(tf.int64, shape=(None))
    labels_gluten = tf.placeholder(tf.int64, shape=(None))
    labels_ibd = tf.placeholder(tf.int64, shape=(None))
    labels_lactose = tf.placeholder(tf.int64, shape=(None))
    labels_quino = tf.placeholder(tf.int64, shape=(None))
    eps = tf.random_normal([batch_size, eps_dim])

    n_samples=4
    U,V,B,DELTA_inv,M,M_gluten,M_ibd,W_gluten,W_ibd = generator(inp_data_dim,inp_cov_dim,latent_dim,rank,keep_prob,n_samples=n_samples)
    U1 = tf.slice(U,[0,0,0],[1,-1,-1])
    V1 = tf.slice(V,[0,0,0],[1,-1,-1])
    M1 = tf.slice(M,[0,0,0],[1,-1,-1])
    A1 = tf.matmul(U1,V1)
    A1 = A1*M1
    B1 = tf.slice(B,[0,0,0],[1,-1,-1])
    DELTA_inv1 = tf.slice(DELTA_inv, [0,0],[1,-1])
    M_gluten = tf.slice(M_gluten,[0,0],[1,-1])
    M_ibd = tf.slice(M_ibd,[0,0],[1,-1])
    W_gluten = tf.slice(W_gluten,[0,0],[1,-1])
    W_ibd = tf.slice(W_ibd,[0,0],[1,-1])
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
    #logits, closs2 = classifier_cnp(z,labels_cnp,W_cnp,M_cnp,reuse=False)
    ##logits = tf.Print(logits,[logits],message="prob tb")
    #correct_label_pred = tf.equal(tf.argmax(logits,1),labels_cnp)
    #label_acc2 = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))

    #logits, closs3 = classifier_diabetes(z,labels_diabetes,W_diabetes,M_diabetes,reuse=False)
    #correct_label_pred = tf.equal(tf.argmax(logits,1),labels_diabetes)
    ##correct_label_pred = tf.Print(correct_label_pred,[correct_label_pred],message="Avtive pred train")
    #label_acc3 = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))

    logits, closs4 = classifier_gluten(z,labels_gluten,W_gluten,M_gluten,reuse=False)
    correct_label_pred = tf.equal(tf.argmax(logits,1),labels_gluten)
    label_acc4 = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))

    logits, closs5 = classifier_ibd(z,labels_ibd,W_ibd,M_ibd,reuse=False)
    correct_label_pred = tf.equal(tf.argmax(logits,1),labels_ibd)
    label_acc5 = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))

    #logits, closs6 = classifier_lactose(z,labels_lactose,W_lactose,M_lactose,reuse=False)
    #correct_label_pred = tf.equal(tf.argmax(logits,1),labels_lactose)
    #label_acc6 = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))

    #logits, closs7 = classifier_quino(z,labels_quino,W_quino,M_quino,reuse=False)
    #correct_label_pred = tf.equal(tf.argmax(logits,1),labels_quino)
    #label_acc7 = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))

    closs = tb_coeff*(closs4+closs5)
    label_acc = (label_acc4+label_acc5)/2.0

    # Dual loss
    Td = data_network(z)
    Ti = data_network(z_sampled, reuse=True)
    d_loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Td, labels=tf.ones_like(Td)))
    d_loss_i = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Ti, labels=tf.zeros_like(Ti)))
    dual_loss = d_loss_d+d_loss_i

    # dual loss for theta
    dual_loss_theta, label_acc_adv_theta, q_ratio = cal_theta_adv_loss(A1,B,DELTA_inv,inp_data_dim, inp_cov_dim, latent_dim, rank,keep_prob,n_samples=batch_size)

    #Adversary Accuracy
    correct_labels_adv = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.greater(tf.sigmoid(Td),thresh_adv), tf.int32),1),tf.float32)) + tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.less_equal(tf.sigmoid(Td),thresh_adv), tf.int32),0),tf.float32))
    label_acc_adv = correct_labels_adv/(2.0*batch_size)

    # Primal loss
    t1 = -tf.reduce_mean(Ti)
    t2 = x_post_prob_log
    t5 = logz-logr
    t3 = tf.reduce_mean(q_ratio)
    f_input = tf.squeeze(q_ratio)-Td
    f_input = tf.exp(f_input)
    print("f_input_shape:",f_input.get_shape().as_list())
    t4 = tf.reduce_mean(tf.square(f_input-1))
    KL_neg_r_q = t1
    ELBO = pearson*(t2-t4)+reverse_kl*(t1+t2+t3+t5)
    primal_loss = tf.reduce_mean(-ELBO)

    print("V in main:",V.get_shape().as_list())
    print("U in main:",U.get_shape().as_list())
    print("B in main:",B.get_shape().as_list())
    print("DELTA_inv in main:",DELTA_inv.get_shape().as_list())
    train(z, closs, label_acc_adv_theta)
