import tensorflow as tf
import numpy as np
import pickle as pkl

slim = tf.contrib.slim
ds = tf.contrib.distributions
st = tf.contrib.bayesflow.stochastic_tensor
graph_replace = tf.contrib.graph_editor.graph_replace


""" Hyperparameters """
latent_dim=100
nepoch = 1500
lr = 10**(-4)
batch_size = 160
ntrain = 160
test_batch_size = 52
ntest=52
inp_data_dim = 5000
inp_cov_dim = 7
L = 5 # Number of samples of z2 drawn from the posterior
X_train, C_train = None, None


def standard_normal(shape, **kwargs):
    """Create a standard Normal StochasticTensor."""
    return tf.cast(st.StochasticTensor(
        ds.MultivariateNormalDiag(tf.zeros(shape), tf.ones(shape), **kwargs)),  tf.float32)

def data_network(z, n_layer=2, n_hidden=256, reuse=False):
    """ Calculates the value of log(r(z_2|x)/q(z_2|x))"""
    with tf.variable_scope("data_net", reuse = reuse):
        h = slim.repeat(z, n_layer, slim.fully_connected, n_hidden, activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(0.1))
        out = slim.fully_connected(h, 1, activation_fn=None, weights_regularizer=slim.l2_regularizer(0.1))
    return out

def encoder(x,c, eps, n_layer=2, n_hidden=256, reuse=False):
    with tf.variable_scope("Encoder", reuse = reuse):
        h = tf.concat([x,c,eps], axis=1)
        h = slim.repeat(h, n_layer, slim.fully_connected, n_hidden, activation_fn=tf.nn.elu, weights_regularizer=slim.l2_regularizer(0.1))
        out = slim.fully_connected(h, latent_dim, activation_fn=None, weights_regularizer=slim.l2_regularizer(0.1))
    return out

def cal_loss(r_sample, z_list):
    with tf.variable_scope("Loss"):
        r_sample_list = [] # A list of L (batch_size,K) tensors
        for i in range(L):
            r_sample_list.append(tf.slice(r_sample, [0,i,0], [-1,1,-1]))
        g_r_z = data_network(r_sample_list[0], reuse=False)
        g_r_z_log_list = [g_r_z]
        for i in range(1,L):
            g_r_z = data_network(r_sample_list[0], reuse=True)
            g_r_z_log_list.append(g_r_z - tf.nn.softplus(g_r_z))                
        g_r_z = tf.addn(g_r_z_log_list)/(1.0*L)
        f1 = tf.reduce_mean(g_r_z, axis=0)

        g_z_log_list = []
        g_z_list = []
        for i in range(len(z_list)):
            g_z = data_network(z_list[i], reuse=True)
            g_z_list.append(g_z)
            g_z_log_list.append(-tf.nn.softplus(g_z))
        g_z = tf.addn(g_z_log_list)/(1.0*len(g_z_log_list))
        f2 = tf.reduce_mean(g_z, axis=0)

        si = f1+f2

        g_z = tf.addn(g_z_list)/(1.0*len(g_z_list))
        f1 = tf.reduce_mean(g_z, axis=0)

        f2 = (z_prior*x_prob)/(1e-12 + 1.0*z_prob)
        f2 = tf.log(f2)
        f2 = tf.reduce_mean(f2, axis=0)
        f2 = tf.reduce_mean(f2,axis=0)

        vtp = f1+f2

def train(z):
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith("Loss")]
    tp_var = [var for var in t_vars if var not in d_vars]
    r_loss = tf.losses.get_regularization_loss()
    opt = tf.train.AdamOptimizer(1e-5)
    si_max_op = opt.minimize(-si+r_loss, var_list=d_vars)
    vtp_min_opt = opt.minimize(vtp+r_loss, var_list=tp_var)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    si_list = []
    vtp_list = []
    for i in range(nepoch):
        for j in range(ntrain//batch_size):
            xmb = X_train[j*batch_size:(j+1)*batch_size]
            cmb = C_train[j*batch_size:(j+1)*batch_size]
            sess.run(sample_from_r, feed_dict={x:xmb, c:cmb})
            for k in range(1):
                _,si_loss = sess.run([si_max_op, si], feed_dict={x:xmb, c:cmb})
            for k in range(1):
                _,vtp_loss = sess.run([vtp_min_opt, vtp], feed_dict={x:xmb, c:cmb})
            si_list.append(si_loss)
            vtp_list.append(vtp_loss)
        if i%100 == 0:
            print("epoch:",i," si:",si_list[-1]," vtp:",vtp_list[-1])

    # Training complete
    z_list=[]
    for i in range(20):
        tmp = []
        for j in range(ntrain//batch_size):
            xmb = X_train[j*batch_size:(j+1)*batch_size]
            cmb = C_train[j*batch_size:(j+1)*batch_size]
            z_ = sess.run(z, feed_dict={x:xmb,c:cmb})
            tmp.append(z_)
        train_z = np.stack(tmp, axis=0)
        z_list.append(train_z)
    
    np.save("si_loss.npy", si_list)
    np.save("vtp_loss.npy", si_list)
    np.save("z_train_list.npy",z_list)

    # Test Set
    z_list = []
    eps = standard_normal((test_batch_size, inp_data_dim))
    z = encoder(X_t,C_t,eps,reuse=False)
    z_list = [z]
    for i in range(20):
        z_ = sess.run(z)
        z_list.append(z_)
    np.save("z_test_list.npy",z_list)

if __name__ == "__main__":
    x = tf.placeholder(tf.float32, shape=(batch_size, inp_data_dim))
    c = tf.placeholder(tf.float32, shape=(batch_size, inp_cov_dim))    

    A = tf.Variable(initial_value=np.random.rand(inp_data_dim, latent_dim))
    B = tf.Variable(initial_value=np.random.rand(inp_data_dim, inp_cov_dim))
    DELTA = tf.Variable(initial_value=np.random.rand(inp_data_dim))
    
    """ Define the distribution r(z2|x) and sample from it """
    RA = np.random.rand(latent_dim, inp_data_dim)
    r_sample = tf.Variable(initial_value=np.zeros((batch_size, L, latent_dim)))
    d = np.random.rand(batch_size,latent_dim)
    means = np.matmul(x, np.transpose(RA))
    R = ds.MultivariateNormalDiag(means, np.abs(d))
    s = R.sample(L) # (L,N,K)
    s = tf.transpose(s, [1,0,2])
    sample_from_r = tf.assign(r_sample, s)

    """ Draw samples from posterior q(z2|x) """
    eps = standard_normal((batch_size, inp_data_dim))
    z = encoder(x,eps,reuse=False)
    z_list = [z]
    z_prob_list = [R.prob(z)]
    for i in range(L):
        z = encoder(x, eps, reuse=True)
        z_list.append(z)
        z_prob_list.append(R.prob(z))
    z_prob = tf.stack(z_prob_list, axis=0) #(L,N)
    z_prob = tf.transpose(z_prob, [1,0]) # (N,L)

    """ Define the prior on z2 and evaluate it on posterior """
    z_list_prior_prob = []
    z_prior_ds = ds.MultivariateNormalDiag(tf.zeros((latent_dim)), tf.ones((latent_dim)))
    for i in range(L):
        z_list_prior_prob.append(z_prior_ds.prob(z_list[i]))
    z_prior = tf.stack(z_list_prior_prob, axis=0)
    z_prior = tf.transpose(z_prior, [1,0,2]) # (N,L)

    """ Define the distribution p(x|z2) and sample from it """
    z = tf.stack(z_list, axis=0) # (L,N,K)
    A_T = tf.transpose(A, [1,0])
    A_T = tf.ones((L,latent_dim, inp_data_dim))*A_T # (L,K,D)
    post = ds.MultivariateNormalDiag(tf.matmul(z, A_T)+tf.matmul(c, B, transpose_b=True), tf.ones((L,batch_size, inp_data_dim))*DELTA*DELTA) # (L,N,D)
    x_ = tf.ones((L,batch_size,inp_data_dim))*x
    x_prob = post.prob(x_)
    x_prob = tf.transpose(x_prob, [1,0]) #(N,L)

    si, vtp = cal_loss(r_sample, z_list)