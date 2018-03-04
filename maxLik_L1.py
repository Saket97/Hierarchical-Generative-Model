import tensorflow as tf
import numpy as np
import pickle as pkl
import math

slim = tf.contrib.slim
ds = tf.contrib.distributions
st = tf.contrib.bayesflow.stochastic_tensor
graph_replace = tf.contrib.graph_editor.graph_replace

""" Hyperparameters """
latent_dim=100
nepoch = 20000
lr = 10**(-4)
batch_size = 160
ntrain = 160
test_batch_size = 52
ntest=52
inp_data_dim = 5000
inp_cov_dim = 7
eps_dim = 30
eps_nbasis=32

def load_dataset():
    raw_data = np.load("/opt/data/saket/gene_data/data/mod_total_data.npy")
    cov = np.load("/opt/data/saket/gene_data/data/cov.npy")
    labels = np.load("/opt/data/saket/gene_data/data/data_label.npy")
    inp_data_dim = raw_data.shape[1]
    inp_cov_dim = cov.shape[1]
    m = np.mean(raw_data, axis=0)
    raw_data = raw_data-m
    cov = np.log10(cov+0.1)
    return raw_data[0:ntrain],cov[0:ntrain],labels[0:ntrain],raw_data[ntrain:],cov[ntrain:],labels[ntrain:]

X_train, C_train, L_train, X_test, C_test, L_test = load_dataset()
X_test = X_test.astype(np.float32)
C_test = C_test.astype(np.float32)

def get_zlogprob(z, z_dist):
    if z_dist == "gauss":
        logprob = -0.5 * tf.reduce_sum(z*z  + np.log(2*np.pi), [1])
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
        h = slim.repeat(z, n_layer, slim.fully_connected, n_hidden, activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(0.1))
        out = slim.fully_connected(h, 1, activation_fn=None, weights_regularizer=slim.l2_regularizer(0.1))
    #out = tf.Print(out, [out], message="data_net_out")
    return tf.squeeze(out)

def encoder(x,c, eps=None, n_layer=2, n_hidden=256, reuse=False):
    with tf.variable_scope("Encoder", reuse = reuse):
        h = tf.concat([x,c], axis=1)
        h = slim.repeat(h, n_layer, slim.fully_connected, n_hidden, activation_fn=tf.nn.elu, weights_regularizer=slim.l2_regularizer(0.1))
        out = slim.fully_connected(h, latent_dim, activation_fn=None, weights_regularizer=slim.l2_regularizer(0.1))
        
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

    return z,Ez,Varz

def train(z):
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith("data_net")]
    tp_var = [var for var in t_vars if var not in d_vars]
    assert(len(tp_var)+len(d_vars) == len(t_vars))
    assert(len(tp_var)>3)
    r_loss = tf.losses.get_regularization_loss()
    primal_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, use_locking=True, beta1=0.5)
    dual_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, use_locking=True, beta1=0.5)

    primal_train_op = primal_optimizer.minimize(primal_loss+r_loss, var_list=tp_var)
    adversary_train_op = dual_optimizer.minimize(dual_loss+r_loss, var_list=d_vars)
    train_op = tf.group(primal_train_op, adversary_train_op)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    si_list = []
    vtp_list = []
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
    np.save("vtp_loss.npy", vtp_list)
    np.save("z_train_list.npy",z_list)

    # Test Set
    z_list = []
    eps = tf.random_normal(tf.stack([eps_nbasis, test_batch_size,eps_dim]))
    z, _, _ = encoder(X_test,C_test,eps,reuse=True)
    z_list = [z]
    for i in range(20):
        z_ = sess.run(z)
        z_list.append(z_)
    np.save("z_test_list.npy",z_list)

if __name__ == "__main__":
    x = tf.placeholder(tf.float32, shape=(batch_size, inp_data_dim))
    c = tf.placeholder(tf.float32, shape=(batch_size, inp_cov_dim))
    z_sampled = tf.random_normal([batch_size, latent_dim])
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
    logr = -0.5 * tf.reduce_sum(z_norm*z_norm + tf.log(z_var) + np.log(2*np.pi), [1])

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
    means = tf.matmul(z,A, transpose_b=True)
    prec = tf.square(DELTA_inv)
    t = (x-means)
    t1 = t*prec*t
    t1 = -0.5*tf.reduce_sum(t1, axis=1)
    t2 = 0.5*tf.reduce_sum(tf.log(1e-3+prec))
    t3 = -latent_dim*0.5*tf.log(2*math.pi)
    x_post_prob_log = t1+t2+t3
    
    # Dual loss
    Td = data_network(z_sampled)
    Ti = data_network(z_norm, reuse=True)
    Td = tf.Print(Td,[Td],message="Td")
    Ti = tf.Print(Ti,[Ti],message="Ti")
    d_loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Td, labels=tf.ones_like(Td)))
    d_loss_i = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Ti, labels=tf.zeros_like(Ti)))
    d_loss_d = tf.Print(d_loss_d, [d_loss_d], message="d_loss_d")
    d_loss_i = tf.Print(d_loss_i, [d_loss_i], message="d_loss_i")
    dual_loss = d_loss_d+d_loss_i

    # Primal loss
    t1 = Ti
    t2 = x_post_prob_log+logz-logr
    ELBO = t1+t2
    primal_loss = tf.reduce_mean(-ELBO)

    train(z)
