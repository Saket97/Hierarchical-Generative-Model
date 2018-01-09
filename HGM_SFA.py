import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from utils.data_gmm import GMM_distribution, sample_GMM, plot_GMM
from utils.data_utils import shuffle, iter_data
from tqdm import tqdm

slim = tf.contrib.slim
ds = tf.contrib.distributions
st = tf.contrib.bayesflow.stochastic_tensor
graph_replace = tf.contrib.graph_editor.graph_replace

""" Parameters """
inp_data_dim = 10 #d
inp_cov_dim = 10 #d'
latent_dim = 5 #k
batch_size = 100
eps_dim = 2
enc_net_hidden_dim = 10
n_samples = 100
n_epoch = 5

""" Dataset """
X_dataset, C_dataset = load_dataset()
XC_dataset = np.concatenate((X_dataset, C_dataset), axis=1)

""" Networks """
def standard_normal(shape, **kwargs):
    """Create a standard Normal StochasticTensor."""
    return tf.cast(st.StochasticTensor(
        ds.MultivariateNormalDiag(mu=tf.zeros(shape), diag_stdev=tf.ones(shape), **kwargs)),  tf.float32)

def decoder_network(z1, z2, c):
    """ Create the decoder network with skip connections. 
    Arguments:
        z1: components of the latent features z which is given as input at layer 1
        z2: components of the latent features z which is given as input as layer 2
        c: covariates matrix

    Return:
        y1, y2: output of the layer 1 and 2
    """
    assert(z1.get_shape().as_list()[0] == z2.get_shape().as_list()[0])
    with tf.variable_scope("decoder", reuse = True):
        A = tf.get_variable("A", shape=(inp_data_dim, latent_dim))
        B = tf.get_variable("B", shape=(inp_data_dim, inp_cov_dim))
        DELTA = tf.get_variable("DELTA", shape=(inp_data_dim)) # It is a diagonal matrix
        DELTA = tf.diag(DELTA)
        
        y1 = tf.matmul(z2, A, transpose_b=True) + tf.matmul(c, B, transpose_b=True)
        y2 = y1 + tf.matmul(z1, DELTA, transpose_b=True)
    return y1, y2

def encoder_network(x, c, latent_dim, n_layer, z1_dim, z2_dim, eps_dim):
    with tf.variable_scope("encoder", reuse = True):
        eps1 = standard_normal([x.get_shape().as_list()[0], eps_dim], name="eps1") * 1.0 # (batch_size, eps_dim)
        eps2 = standard_normal([x.get_shape().as_list()[0], eps_dim], name="eps2") * 1.0 # (batch_size, eps_dim)

        h = tf.concat([x, c, eps], 1)
        h = slim.repeat(h, n_layer, slim.fully_connected, latent_dim)
        z1 = slim.fully_connected(h, z1_dim)

        h = tf.concat([x, c, eps2, z1], axis=1)
        h = slim.repeat(h, n_layer, slim.fully_connected, latent_dim)
        z2 = slim.fully_connected(h, z2_dim)
    return z1, z2

def data_network(x, z, n_layer=2, n_hidden=256):
    """ The network to approximate the function g_si(x,z) whose optimal value will give w(x,z)
    Arguments:
        x: Data matrix of dimension (batch_size, inp_data_dim)
        z: Latent features of dimension (batch_size, z1+z2)
    Return:
        Evaluation of g_si(x,z) which is a scalar
        """
    with tf.variable_scope("data_network", reuse = True)
        h = tf.concat([x,z], axis=1)
        h = slim.repeat(h, n_layer, slim.fully_connected, n_hidden)
        h = slim.fully_connected(h, 1)
    return h

def cal_maximising_quantity(z_sample, x_sample, z, x):
    """ Expression which needs to be maximised for Si """
    with tf.variable_scope("Si", reuse=True):
        g_si = data_network(x_sample, z_sample)
        g_si_sig = tf.nn.sigmoid(g_si)
        f = tf.log(1-g_si_sig)
        assert(f.get_shape() == (x_sample.get_shape().as_list()[0], 1))
        first_term = tf.truediv(tf.reduce_sum(f), tf.shape(f)[0])

        g_si = data_network(x,z)
        g_si_sig = tf.nn.sigmoid(g_si)
        f = tf.log(g_si)
        second_term = tf.truediv(tf.reduce_sum(f), tf.shape(f)[0])

        final = first_term+second_term
    return final

def cal_loss(x_sample, z_sample, x, z):
    with tf.variable_scope("loss", reuse=True):
        w_x_z = data_network(x,z)
        f1 = tf.truediv(tf.reduce_sum(w_x_z), tf.shape(w_x_z)[0])

        w_x_z = data_network(x_sample, z_sample)
        f2 = tf.truediv(w_x_z, tf.shape(w_x_z)[0])

        final = f1-f2
    return final

""" Construct model """
tf.reset_default_graph()

x = tf.placeholder(tf.float32, shape=(batch_size, inp_data_dim))
c = tf.placeholder(tf.float32, shape=(batch_size, inp_cov_dim))

z1, z2 = encoder_network(x, c, enc_net_hidden_dim, 5, inp_data_dim, latent_dim, eps_dim)
y1, y2 = decoder_network(z1, z2, c)
z = tf.concat([z1,z2], axis=1)

MVN = ds.MultivariateNormalDiag(tf.zeros((batch_size, latent_dim+inp_data_dim)), tf.ones((batch_size, latent_dim+inp_data_dim)))
z_sample = MVN.sample(n_samples)
z1_sample = tf.slice(z_sample, [0, 0], [-1, inp_data_dim])
z2_sample = tf.slice(z_sample, [0, inp_data_dim], [-1, latent_dim])
x_sample = decoder_network(z1_sample, z2_sample, c)

si_net_maximise = cal_maximising_quantity(z_sample, x_sample, z, x)
theta_phi_minimise = cal_loss(x_sample, z_sample, x, z)

t_vars = tf.trainable_variables()
svars = [var for var in t_vars if var.name.startswith("Si")]
dnvars = [var for var in t_vars if var.naem.startswith("data_network")]
evars = [var for var in t_vars if var.naem.startswith("encoder")]
dvars = [var for var in t_vars if var.naem.startswith("decoder")]
lvars = [var for var in t_vars if var.naem.startswith("loss")]

opt = tf.train.AdamOptimizer(1e-3, beta1=0.5)
train_si = opt.minimize(-si_net_maximise, var_list=svars+dnvars)
train_t_p = opt.minimize(theta_phi_minimise, var_list=evars+dvars+lvars)

""" Training """
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epoch):
        XC_dataset = np.random.shuffle(XC_dataset)
        X_dataset = XC_dataset[:][0:inp_data_dim]
        C_dataset = XC_dataset[:][inp_data_dim:]

        for i in range(np.shape(X_dataset)[0]/batch_size):
            xmb = X_dataset[i*batch_size:(i+1)*batch_size]
            cmb = C_dataset[i*batch_size:(i+1)*batch_size]

            for _ in range(1):
                si_loss, _ = sess.run([si_net_maximise, train_si], feed_dict={x:xmb, c:cmb})
            for _ in range(1):
                t_loss, _ = sess.run([theta_phi_minimise, train_t_p], feed_dict={x:xmb, c:cmb})
