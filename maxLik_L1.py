import tensorflow as tf
import numpy as np
import pickle as pkl
import math

slim = tf.contrib.slim
ds = tf.contrib.distributions
st = tf.contrib.bayesflow.stochastic_tensor
graph_replace = tf.contrib.graph_editor.graph_replace

""" Hyperparameters """
latent_dim=60
nepoch = 55000
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

def get_zlogprob(z, z_dist):
    if z_dist == "gauss":
        logprob = -0.5 * tf.reduce_sum(z*z  + latent_dim*np.log(2*np.pi), [1])
    elif z_dist == "uniform":
        logprob = 0.
    else:
        raise ValueError("Invalid parameter value for `z_dist`.")
    return logprob

def standard_normal(shape, **kwargs):
    """Create a standard Normal StochasticTensor."""
    return tf.cast(st.StochasticTensor(
        ds.MultivariateNormalDiag(tf.zeros(shape), tf.ones(shape), **kwargs)),  tf.float32)

def data_network(z, n_layer=2, n_hidden=256, reuse=False):
    """ Calculates the value of log(r(z_2|x)/q(z_2|x))"""
    with tf.variable_scope("data_net", reuse = reuse):
        h = slim.repeat(z, n_layer, slim.fully_connected, n_hidden, activation_fn=tf.nn.elu, weights_regularizer=slim.l2_regularizer(0.01))
        out = slim.fully_connected(h, 1, activation_fn=None, weights_regularizer=slim.l2_regularizer(0.01))
    #out = tf.Print(out, [out], message="data_net_out")
    return tf.squeeze(out)

def encoder(x,c, eps=None, n_layer=2, n_hidden=256, reuse=False):
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

def classifier(x_input,labels, reuse=False):
    with tf.variable_scope("Classifier", reuse=reuse):
        logits = slim.fully_connected(x_input, 3, activation_fn=None, weights_regularizer=slim.l2_regularizer(0.1))
        cl_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    return logits, tf.reduce_mean(cl_loss)

def train(z, closs):
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith("data_net")]
    c_vars = [var for var in t_vars if var.name.startswith("Classifier")]
    tp_var = [var for var in t_vars if var not in d_vars+c_vars]
    assert(len(tp_var)+len(d_vars)+len(c_vars) == len(t_vars))
    assert(len(tp_var)>3)
    r_loss = tf.losses.get_regularization_loss()
    r_loss_clf = tf.losses.get_regularization_loss(scope="Classifier")
    r_loss -= r_loss_clf
    r_loss += l1_regulariser(A)
    r_loss += l1_regulariser(B)
    primal_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, use_locking=True, beta1=0.5)
    dual_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, use_locking=True, beta1=0.5)
    classifier_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3, use_locking=True)

    primal_train_op = primal_optimizer.minimize(primal_loss+r_loss, var_list=tp_var)
    adversary_train_op = dual_optimizer.minimize(dual_loss+r_loss, var_list=d_vars)
    train_op = tf.group(primal_train_op, adversary_train_op)
    clf_train_op = classifier_optimizer.minimize(closs+r_loss_clf, var_list=c_vars)
    
    # Test Set Graph
    eps = tf.random_normal(tf.stack([eps_nbasis, test_batch_size,eps_dim]))
    z_test, _, _ = encoder(X_test,C_test,eps,reuse=True)
    means = tf.matmul(z_test,A, transpose_b=True)+tf.matmul(C_test,B, transpose_b=True)
    prec = tf.square(DELTA_inv)
    t = (X_test-means)
    t1 = t*prec*t
    t1 = -0.5*tf.reduce_sum(t1, axis=1)
    t2 = 0.5*tf.reduce_sum(tf.log(1e-3+prec))
    t3 = -inp_data_dim*0.5*tf.log(2*math.pi)
    x_post_prob_log_test = t1+t2+t3
    x_post_prob_log_test = tf.reduce_mean(x_post_prob_log_test)
   
    saver = tf.train.Saver()
    sess = tf.Session()
    #saver.restore(sess,"/opt/data/saket/model.ckpt-54000")
    sess.run(tf.global_variables_initializer())
    si_list = []
    vtp_list = []
    post_test_list = []
    for i in range(nepoch):
        for j in range(ntrain//batch_size):
            xmb = X_train[j*batch_size:(j+1)*batch_size]
            cmb = C_train[j*batch_size:(j+1)*batch_size]
            # sess.run(sample_from_r, feed_dict={x:xmb, c:cmb})
            sess.run(train_op, feed_dict={x:xmb,c:cmb})
            si_loss,vtp_loss = sess.run([dual_loss, primal_loss], feed_dict={x:xmb,c:cmb})
            si_list.append(si_loss)
            vtp_list.append(vtp_loss)
        if i%100 == 0:
            Td_,KL_neg_r_q_,x_post_prob_log_,logz_,logr_,d_loss_d_,d_loss_i_,label_acc_adv_ = sess.run([Td,KL_neg_r_q,x_post_prob_log,logz,logr,d_loss_d,d_loss_i,label_acc_adv], feed_dict={x:xmb,c:cmb})
            print("epoch:",i," si:",si_list[-1]," vtp:",vtp_list[-1]," -KL_r_q:",KL_neg_r_q_, " x_post:",x_post_prob_log_," logz:",logz_," logr:",logr_, \
            " d_loss_d:",d_loss_d_, " d_loss_i:",d_loss_i_, " adv_accuracy:",label_acc_adv_)
        if i%1000 == 0:
            x_post_test = sess.run(x_post_prob_log_test)
            post_test_list.append(x_post_test)
            print("test set p(x|z):",x_post_test, "Td:",Td_)
            path = saver.save(sess,"/opt/data/saket/model.ckpt",i)
            print("Model saved at ",path)
    # Training complete
    z_list=[]
    clf_loss_list = []
    for i in range(n_clf_epoch):
        tmp = []
        for j in range(ntrain//batch_size):
            xmb = X_train[j*batch_size:(j+1)*batch_size]
            cmb = C_train[j*batch_size:(j+1)*batch_size]
            sess.run(clf_train_op, feed_dict={x:xmb,c:cmb,labels:L_train[j*batch_size:(j+1)*batch_size]})           
            z_ = sess.run(z, feed_dict={x:xmb,c:cmb})
            cl_loss_, label_acc_ = sess.run([closs, label_acc], feed_dict={x:xmb,c:cmb,labels:L_train[j*batch_size:(j+1)*batch_size]})
            tmp.append(z_)
        if i%100 == 0:
            print("epoch:",i," closs:",cl_loss_," train_acc:",label_acc_)
        train_z = np.stack(tmp, axis=0)
        if i < 250:
            z_list.append(train_z)
        clf_loss_list.append((cl_loss_, label_acc_))
    
    A_,B_,DELTA_inv_ = sess.run([A,B,DELTA_inv])
    np.save("si_loss1.npy", si_list)
    np.save("vtp_loss1.npy", vtp_list)
    np.save("x_post_list1.npy",post_test_list)
    np.save("z_train_list1.npy",z_list)
    np.save("A1.npy",A_)
    np.save("B1.npy",B_)
    np.save("delta_inv1.npy",DELTA_inv_)
    np.save("clf_loss_list1.npy",clf_loss_list)

    # Test Set
    z_list = []
    logits_test, closs_test = classifier(z_test,labels,reuse=True)
    prob_test = tf.nn.softmax(logits_test)
    correct_label_pred_test = tf.equal(tf.argmax(logits_test,1),labels)
    label_acc_test = tf.reduce_mean(tf.cast(correct_label_pred_test, tf.float32))        
    label_acc_ = sess.run(label_acc_test, feed_dict={labels:L_test})
    print("Test Set label Accuracy:", label_acc_)
    z_list = []
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
    

    with tf.variable_scope("Decoder"):
        A = tf.Variable(initial_value=np.random.rand(inp_data_dim, latent_dim).astype(np.float32))
        B = tf.Variable(initial_value=np.random.rand(inp_data_dim, inp_cov_dim).astype(np.float32))
        DELTA_inv = tf.Variable(initial_value=np.random.rand(inp_data_dim).astype(np.float32))

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
    # Evaluating r(z|x)
    # t = (z-R_mean)
    # t1 = t*tf.exp(R_log_prec)*t
    # t1 = -0.5*tf.reduce_sum(t1, axis=1)
    # t2 = 0.5*tf.reduce_sum(R_log_prec, axis=1)
    # t3 = -latent_dim*0.5*tf.log(2*math.pi)
    # r_prob_log = t1+t2+t3

    # # Evaluating prior on the posterior samples
    # t = z
    # t1 = t*t
    # t1 = -0.5*tf.reduce_sum(t1, axis=1)
    # t2 = 0
    # t3 = -latent_dim*0.5*tf.log(2*math.pi)
    # prior_prob_log = t1+t2+t3

    # Evaluating p(x|z)
    means = tf.matmul(z,A, transpose_b=True)+tf.matmul(c,B, transpose_b=True)
    prec = tf.square(DELTA_inv)
    t = (x-means)
    t1 = t*prec*t
    t1 = -0.5*tf.reduce_sum(t1, axis=1)
    t2 = 0.5*tf.reduce_sum(tf.log(1e-3+prec))
    t3 = -inp_data_dim*0.5*tf.log(2*math.pi)
    x_post_prob_log = t1+t2+t3
    x_post_prob_log = tf.reduce_mean(x_post_prob_log)

    # Classifier
    logits, closs = classifier(z,labels,reuse=False)
    correct_label_pred = tf.equal(tf.argmax(logits,1),labels)
    label_acc = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))
    
    # Dual loss
    Td = data_network(z_norm)
    Ti = data_network(z_sampled, reuse=True)
    #Td = tf.Print(Td,[Td],message="Td")
    #Ti = tf.Print(Ti,[Ti],message="Ti")
    d_loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Td, labels=tf.ones_like(Td)))
    d_loss_i = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Ti, labels=tf.zeros_like(Ti)))
    #d_loss_d = tf.Print(d_loss_d, [d_loss_d], message="d_loss_d")
    #d_loss_i = tf.Print(d_loss_i, [d_loss_i], message="d_loss_i")
    dual_loss = d_loss_d+d_loss_i
    
    #Adversary Accuracy
    correct_labels_adv = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.greater(tf.sigmoid(Td),thresh_adv), tf.int32),1),tf.float32)) + tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.less_equal(tf.sigmoid(Td),thresh_adv), tf.int32),0),tf.float32))
    label_acc_adv = correct_labels_adv/(2.0*batch_size)

    # Primal loss
    t1 = -tf.reduce_mean(Td)
    t2 = x_post_prob_log+logz-logr
    KL_neg_r_q = t1
    ELBO = t1+t2
    primal_loss = tf.reduce_mean(-ELBO)

    train(z, closs)
