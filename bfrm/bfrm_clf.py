import tensorflow as tf
import numpy as np
import sklearn.metrics as skm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num',type=int,default=10)
FLAGS, unparsed = parser.parse_known_args()

nepoch = 4001
lr = 10**(-4)
batch_size = 144
ntrain = 288
test_batch_size = 32
ntest=32
fold_size = 32
inp_data_dim = 60

A = np.loadtxt("mA.txt") # (D X nfactor)
A = A[:,1:]
X = np.load("/opt/data/saket/gene_data/data/mod_gauss_data.npy") # (N X D)
X = X[0:10*fold_size]
m = np.mean(X, axis=0)
X = (X-m)
Psi = np.loadtxt("mPsi.txt") # (D)
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def _get_samples_mvn(mu, sigma, n=1):
    """ 
    Returns n samples from multivariate normal distribution

    Return: A matix of shape (n x shape of mu)
    """
    L = np.linalg.cholesky(sigma)
    t = np.random.normal(size=(n,mu.shape[0]))
    t = np.matmul(L,t.T).T+mu
    return t
    t = np.random.multivariate_normal(mu,sigma, n)
    print("Is positive definite:",is_pos_def(sigma))
    return t

def get_samples_mvn(mu, sigma, n=1):
    """
        Get n samples from each of the MVNs defined by the rows of mu with covariance sigma
    """
    samples = []
    for i in range(mu.shape[0]):
        t = _get_samples_mvn(mu[i], sigma, n)
        samples.append(t)

    return np.stack(samples, axis=0)

def get_input(A,X,Psi):
    
    Sigma_inv = np.matmul(A.T, (1.0/np.expand_dims(Psi+1e-5,axis=1))*A)
    Sigma = np.linalg.inv(Sigma_inv)
    
    mu = np.matmul(Sigma.T, np.matmul(A.T, (1.0/np.expand_dims(Psi+1e-5,axis=1))*X.T )  ) # (nfactor x nfactor) (nfactor,D) (D x N) = (nfactor x N)
    mu = mu.T # (N x nfactor)
    
    return mu, Sigma

def load_dataset(A,X,Psi):
    mu, Sigma = get_input(A,X,Psi)
    raw_data = np.mean(get_samples_mvn(mu,Sigma),axis=1)
    labels = np.load("/opt/data/saket/gene_data/data/hiv.npy")
    labels_tb = np.load("/opt/data/saket/gene_data/data/tb.npy")
    labels = np.squeeze(labels)
    labels_tb = np.squeeze(labels_tb)
    labels = labels[0:fold_size*10]
    labels_tb = labels_tb[0:fold_size*10]
    
    n = FLAGS.num
    start = (n-1)*fold_size
    end = n*fold_size
    L_test_ld = labels[start:end]
    Ltb_test1_ld = labels_tb[start:end]
    mask = np.ones(labels.shape[0],dtype=bool)
    mask[start:end] = False
    L_train_ld = labels[mask,...]
    Ltb_train_ld = labels_tb[mask,...]
    raw_data_ld = raw_data[mask,...]
    return raw_data_ld,L_train_ld,Ltb_train_ld,L_test_ld,Ltb_test1_ld,mu[start:end],Sigma

X_train, L_train, Ltb_train, L_test, Ltb_test1, mu_test, Sigma = load_dataset(A,X,Psi)
print ("X_train:",X_train)
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

def Classifier_active_tb(x_input, labels, reuse=False):
    with tf.variable_scope("Classifier_active_tb", reuse=reuse):
        #x_input += tf.random_normal(x_input.get_shape().as_list())
        b = tf.get_variable("bias",shape=[1],dtype=tf.float32)
        W = tf.get_variable("weight", shape=[1,inp_data_dim], dtype=tf.float32)
        
        labels = tf.cast(labels,tf.float32)
        
        logits = tf.matmul(x_input,W,transpose_b=True)+b
        probs  = tf.sigmoid(logits) # (N,1)
        loss   = tf.expand_dims(labels,axis=1)*tf.log(probs+1e-5) + tf.expand_dims(1-labels,axis=1)*tf.log(1-probs+1e-5)
        cl_loss = -tf.reduce_mean(loss)
        probs = tf.squeeze(probs)
    return tf.stack([1-probs,probs],axis=1), cl_loss

def Classifier_latent_tb(x_input, labels, reuse=False):
    with tf.variable_scope("Classifier_latent_tb", reuse=reuse):
        #x_input += tf.random_normal(x_input.get_shape().as_list())
        b = tf.get_variable("bias",shape=[1],dtype=tf.float32)
        W = tf.get_variable("weight", shape=[1,inp_data_dim], dtype=tf.float32)
        
        labels = tf.cast(labels,tf.float32)
        
        logits = tf.matmul(x_input,W,transpose_b=True)+b
        probs  = tf.sigmoid(logits) # (N,1)
        loss   = tf.expand_dims(labels,axis=1)*tf.log(probs+1e-5) + tf.expand_dims(1-labels,axis=1)*tf.log(1-probs+1e-5)
        cl_loss = -tf.reduce_mean(loss)
        probs = tf.squeeze(probs)
    return tf.stack([1-probs,probs],axis=1), cl_loss

def Classifier_tb(x_input, labels, reuse=False):
    with tf.variable_scope("Classifier_tb", reuse=reuse):
        #x_input += tf.random_normal(x_input.get_shape().as_list())
        b = tf.get_variable("bias",shape=[1],dtype=tf.float32)
        W = tf.get_variable("weight", shape=[1,inp_data_dim], dtype=tf.float32)
        
        labels = tf.cast(labels,tf.float32)
        
        logits = tf.matmul(x_input,W,transpose_b=True)+b
        probs  = tf.sigmoid(logits) # (N,1)
        loss   = tf.expand_dims(labels,axis=1)*tf.log(probs+1e-5) + tf.expand_dims(1-labels,axis=1)*tf.log(1-probs+1e-5)
        cl_loss = -tf.reduce_mean(loss)
        probs = tf.squeeze(probs)
    return tf.stack([1-probs,probs],axis=1), cl_loss

def Classifier_hiv(x_input, labels, reuse=False):
    with tf.variable_scope("Classifier_hiv", reuse=reuse):
        #x_input += tf.random_normal(x_input.get_shape().as_list())
        b = tf.get_variable("bias",shape=[1],dtype=tf.float32)
        W = tf.get_variable("weight", shape=[1,inp_data_dim], dtype=tf.float32)
        
        labels = tf.cast(labels,tf.float32)
        
        logits = tf.matmul(x_input,W,transpose_b=True)+b
        probs  = tf.sigmoid(logits) # (N,1)
        loss   = tf.expand_dims(labels,axis=1)*tf.log(probs+1e-5) + tf.expand_dims(1-labels,axis=1)*tf.log(1-probs+1e-5)
        cl_loss = -tf.reduce_mean(loss)
        probs = tf.squeeze(probs)
    return tf.stack([1-probs,probs],axis=1), cl_loss

def train(closs):
    t_vars = tf.trainable_variables()
    lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in t_vars if 'bias' not in v.name ]) * 0.1

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, use_locking=True, beta1=0.5)
    print("tvars:",t_vars)
    #opt_grad = optimizer.compute_gradients(closs, var_list=t_vars)
    #capped_g_grad = []
    #for grad,var in opt_grad:
    #    if grad is not None:
    #        capped_g_grad.append((tf.clip_by_value(grad,-0.1,0.1),var))
    #train_op = optimizer.apply_gradients(capped_g_grad)
    
    train_op = optimizer.minimize(closs+lossL2, var_list=t_vars)
    # Test Set Graph
    prob_test1, closs_test1 = Classifier_hiv(X_test, L_test, reuse=True)
    prob_test2, closs_test2 = Classifier_tb(X_test, Ltb_test, reuse=True)
    prob_test3, closs_test3 = Classifier_active_tb(X_test, Lac_test, reuse=True)
    prob_test4, closs_test4 = Classifier_latent_tb(X_test, Lla_test, reuse=True)
    #print("prob_test4 shape:",prob_test4.get_shape().as_list())
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    mu, Sigma = get_input(A,X,Psi)
    auc = []
    closs_list = []
    for i in range(nepoch):
        raw_data = np.mean(get_samples_mvn(mu,Sigma),axis=1)
        n = FLAGS.num
        start = (n-1)*fold_size
        end = n*fold_size
        mask = np.ones(raw_data.shape[0],dtype=bool)
        mask[start:end] = False
        X_train = raw_data[mask,...]
        for j in range(ntrain//batch_size):

            xmb = X_train[j*batch_size:(j+1)*batch_size]

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

            lhiv_mb = L_train[j*batch_size:(j+1)*batch_size]

            sess.run(train_op, feed_dict={x:xmb,labels_active:lacmb, labels_hiv:lhiv_mb, labels_latent:llamb, labels_tb:ltbmb})
        
        closs_ = sess.run(closs, feed_dict={x:xmb,labels_active:lacmb, labels_hiv:lhiv_mb, labels_latent:llamb, labels_tb:ltbmb})
        closs_list.append(closs_)
        if i%100 == 0:
            print ("Epoch:%d closs:%f"%(i,closs_))

        if i%100 == 0:
            # test_lik_list = []
            test_prob_hiv = []
            test_prob_tb = []
            test_prob_ac = []
            test_prob_la = []
            for k in range(100):
                # test_lik = sess.run(x_post_prob_log_test)
                # test_lik_list.append(test_lik)
                lt_hiv,lt_tb,lt_ac,lt_la = sess.run([prob_test1, prob_test2,prob_test3,prob_test4], feed_dict = {X_test:np.mean(get_samples_mvn(mu_test,Sigma),axis=1)})
                test_prob_hiv.append(lt_hiv)
                test_prob_tb.append(lt_tb)
                test_prob_ac.append(lt_ac)
                test_prob_la.append(lt_la)
            avg_test_prob_hiv = np.mean(np.stack(test_prob_hiv,axis=0),axis=0)
            avg_test_prob_tb = np.mean(np.stack(test_prob_tb,axis=0),axis=0)
            avg_test_prob_ac = np.mean(np.stack(test_prob_ac,axis=0), axis=0)
            avg_test_prob_la = np.mean(np.stack(test_prob_la,axis=0),axis=0)

            if i==1000:
                print("avg_test_hiv:",avg_test_prob_hiv)
                print("ag_test_prob_tb:",avg_test_prob_tb)
                print("avg_test_prob_ac:",avg_test_prob_ac)
                print("ag_test_prob_la:",avg_test_prob_la)

            avg_test_acc_hiv = np.mean((np.argmax(avg_test_prob_hiv,axis=1)==L_test))
            avg_test_acc_tb = np.mean((np.argmax(avg_test_prob_tb,axis=1)==Ltb_test))
            avg_test_acc_ac = np.mean((np.argmax(avg_test_prob_ac,axis=1)==Lac_test))
            avg_test_acc_la = np.mean((np.argmax(avg_test_prob_la,axis=1)==Lla_test))
            print("Average Test Set Accuracy HIV:",avg_test_acc_hiv)
            print("Average Test Set Accuracy TB:",avg_test_acc_tb)
            print("Average Test Set Accuracy Active TB:",avg_test_acc_ac)
            print("Average Test Set Accuracy Latent TB:",avg_test_acc_la)

            tmp = np.zeros((L_test.shape[0],2))
            t = L_test.astype(np.int32)
            tmp[np.arange(L_test.shape[0]),t] = 1
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
            
            auc.append([auc_hiv,auc_tb,auc_ac,auc_la])

    np.save("bfrm_auc%d.npy"%(FLAGS.num),auc)
    np.save("bfrm_closs%d.npy"%(FLAGS.num),closs_list)         
    
if __name__== "__main__":
    
    x = tf.placeholder(tf.float32, shape=(batch_size, inp_data_dim))
    X_test = tf.placeholder(tf.float32, shape=(None, inp_data_dim))

    labels_hiv = tf.placeholder(tf.int64, shape=(None))
    labels_tb = tf.placeholder(tf.int64, shape=(None))
    labels_active = tf.placeholder(tf.int64, shape=(None))
    labels_latent = tf.placeholder(tf.int64, shape=(None))

    logits, closs1 = Classifier_hiv(x,labels_hiv,reuse=False)
    logits, closs2 = Classifier_active_tb(x, labels_active, reuse=False)
    logits, closs3 = Classifier_latent_tb(x, labels_latent, reuse=False)
    logits, closs4 = Classifier_tb(x, labels_tb, reuse=False)

    closs = closs1+closs2+closs3+closs4

    train(closs)
