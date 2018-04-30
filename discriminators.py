import tensorflow as tf
import numpy as np

slim = tf.contrib.slim
ds = tf.contrib.distributions
st = tf.contrib.bayesflow.stochastic_tensor
graph_replace = tf.contrib.graph_editor.graph_replace


def variable_summaries(var, name=None):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('%s mean'%name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('%s stddev'%name, stddev)
        tf.summary.scalar('%s max'%name, tf.reduce_max(var))
        tf.summary.scalar('%s min'%name, tf.reduce_min(var))
        tf.summary.histogram('%s histogram'%name, var)

def get_zlogprob(z, z_dist):
    if z_dist == "gauss":
        logprob = -0.5 * tf.reduce_sum(z*z  + np.log(2*np.pi), [1])
    elif z_dist == "uniform":
        logprob = 0.
    else:
        raise ValueError("Invalid parameter value for `z_dist`.")
    return logprob

def cloglog(x):
    #return tf.sigmoid(20*x)
    return 1-tf.exp(-tf.exp(5*x))

def sample_gumbel(shape, eps=1e-20):
    """ Sample from Gumbel(0,1)"""
    U = tf.random_uniform(shape, minval=0,maxval=1)
    return -tf.log(-tf.log(U+eps) + eps)

def gumble_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution """
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y/tf.expand_dims(temperature, axis=1))

def gumbel_softmax(logits, temperature, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
            logits: [batch_size, n_class] unnormalized log-probs
            temperature: non-negative scalar
            hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
            [batch_size, n_class] sample from the Gumbel-Softmax distribution.
            If hard=True, then the returned sample will be one-hot, otherwise it will
            be a probabilitiy distribution that sums to 1 across classes
    """

    y = gumble_softmax_sample(logits, temperature)
    return y

def standard_normal(shape, **kwargs):
    """Create a standard Normal StochasticTensor."""
    return tf.cast(st.StochasticTensor(
        ds.MultivariateNormalDiag(tf.zeros(shape), tf.ones(shape), **kwargs)),  tf.float32)

def lrelu(x, leak=0.2, name="lrelu"):
    """ Leaky relu activation function """
    return tf.maximum(x, leak*x)

def data_network(z, n_layer=1, n_hidden=256, reuse=False):
    """ Calculates the value of log(r(z_2|x)/q(z_2|x))"""
    with tf.variable_scope("data_net", reuse = reuse):
        h = slim.repeat(z, n_layer, slim.fully_connected, n_hidden, activation_fn=tf.nn.elu, weights_regularizer=slim.l2_regularizer(0.01))
        out = slim.fully_connected(h, 1, activation_fn=None, weights_regularizer=slim.l2_regularizer(0.01))
    #out = tf.Print(out, [out], message="data_net_out")
    return tf.squeeze(out)

def U_ratio(U, inp_data_dim, rank, n_layer=2, n_hidden=128, reuse=False):
    """ U:(n_samples, inp_data_dim, rank) """
    with tf.variable_scope("U", reuse=reuse):
        U = tf.reshape(U, [-1, inp_data_dim*rank])
        h = slim.repeat(U,n_layer,slim.fully_connected,n_hidden,activation_fn=lrelu,weights_regularizer=slim.l2_regularizer(0.1))
        h = slim.fully_connected(h,1,activation_fn=None,weights_regularizer=slim.l2_regularizer(0.1))
        variable_summaries(h,name="U_ratio")
    return h

def V_ratio(U, latent_dim, rank, n_layer=2, n_hidden=128, reuse=False):
    """ U:(n_samples, inp_data_dim, rank) """
    with tf.variable_scope("V", reuse=reuse):
        U = tf.reshape(U, [-1, latent_dim*rank])
        h = slim.repeat(U,n_layer,slim.fully_connected,n_hidden,activation_fn=lrelu,weights_regularizer=slim.l2_regularizer(0.1))
        h = slim.fully_connected(h,1,activation_fn=None,weights_regularizer=slim.l2_regularizer(0.1))
        variable_summaries(h,name="V_ratio")
    return h

def B_ratio(U, inp_data_dim, inp_cov_dim, n_layer=2, n_hidden=128, reuse=False):
    """ U:(n_samples, inp_data_dim, rank) """
    with tf.variable_scope("B", reuse=reuse):
        U = tf.reshape(U, [-1, inp_data_dim*inp_cov_dim])
        h = slim.repeat(U,n_layer,slim.fully_connected,n_hidden,activation_fn=lrelu,weights_regularizer=slim.l2_regularizer(0.1))
        h = slim.fully_connected(h,1,activation_fn=None,weights_regularizer=slim.l2_regularizer(0.1))
        variable_summaries(h,name="B_ratio")
    return h

def D_ratio(D, n_layer=2, n_hidden=128, reuse=False):
    with tf.variable_scope("del", reuse=reuse):
        h = slim.repeat(D,n_layer, slim.fully_connected,n_hidden,activation_fn=lrelu,weights_regularizer=slim.l2_regularizer(0.1))
        h = slim.fully_connected(h,1,activation_fn=None,weights_regularizer=slim.l2_regularizer(0.1))
        variable_summaries(h,name="D_ratio")
    return h

def M_ratio(M,  inp_data_dim, latent_dim,n_layer=2, n_hidden=128, reuse=False):
    with tf.variable_scope("M", reuse=reuse):
        M = tf.reshape(M, [-1, inp_data_dim*latent_dim])
       # M = M+tf.random_normal(tf.shape(M))
        variable_summaries(M,name="M_ratio_input")
        #M = tf.Print(M,[M],message="M input")
        h = slim.repeat(M,n_layer, slim.fully_connected,n_hidden,activation_fn=lrelu,weights_regularizer=slim.l2_regularizer(0.1))
        h = slim.fully_connected(h,1,activation_fn=None,weights_regularizer=slim.l2_regularizer(0.1))
        #h = tf.Print(M,[M],message="M ratio network output...")
        variable_summaries(h,name="M_ratio")
   
    return h

def Mtb_ratio(M, n_layer=2, n_hidden=32, reuse=False):
    with tf.variable_scope("Mtb", reuse=reuse):
        h = slim.repeat(M,n_layer, slim.fully_connected,n_hidden,activation_fn=lrelu,weights_regularizer=slim.l2_regularizer(0.1))
        h = slim.fully_connected(h,1,activation_fn=None,weights_regularizer=slim.l2_regularizer(0.1))
   
    return h

def Mactive_ratio(M, n_layer=2, n_hidden=32, reuse=False):
    with tf.variable_scope("Mactive", reuse=reuse):
        h = slim.repeat(M,n_layer, slim.fully_connected,n_hidden,activation_fn=lrelu,weights_regularizer=slim.l2_regularizer(0.1))
        h = slim.fully_connected(h,1,activation_fn=None,weights_regularizer=slim.l2_regularizer(0.1))
    return h

def Mlatent_ratio(M, n_layer=2, n_hidden=32, reuse=False):
    with tf.variable_scope("Mlatent", reuse=reuse):
        h = slim.repeat(M,n_layer, slim.fully_connected,n_hidden,activation_fn=lrelu,weights_regularizer=slim.l2_regularizer(0.1))
        h = slim.fully_connected(h,1,activation_fn=None,weights_regularizer=slim.l2_regularizer(0.1))
    return h

def Mhiv_ratio(M, n_layer=2, n_hidden=32, reuse=False):
    with tf.variable_scope("Mhiv", reuse=reuse):
        h = slim.repeat(M,n_layer, slim.fully_connected,n_hidden,activation_fn=lrelu,weights_regularizer=slim.l2_regularizer(0.1))
        h = slim.fully_connected(h,1,activation_fn=None,weights_regularizer=slim.l2_regularizer(0.1))
    return h

def Wtb_ratio(M, n_layer=1, n_hidden=64, reuse=False):
    with tf.variable_scope("Wtb", reuse=reuse):
        h = slim.repeat(M,n_layer, slim.fully_connected,n_hidden,activation_fn=lrelu,weights_regularizer=slim.l2_regularizer(0.1))
        h = slim.fully_connected(h,1,activation_fn=None,weights_regularizer=slim.l2_regularizer(0.1))
    return h

def Wactive_ratio(M, n_layer=2, n_hidden=64, reuse=False):
    with tf.variable_scope("Wactive", reuse=reuse):
        h = slim.repeat(M,n_layer, slim.fully_connected,n_hidden,activation_fn=lrelu,weights_regularizer=slim.l2_regularizer(0.1))
        h = slim.fully_connected(h,1,activation_fn=None,weights_regularizer=slim.l2_regularizer(0.1))
    return h

def Wlatent_ratio(M, n_layer=2, n_hidden=64, reuse=False):
    with tf.variable_scope("Wlatent", reuse=reuse):
        h = slim.repeat(M,n_layer, slim.fully_connected,n_hidden,activation_fn=lrelu,weights_regularizer=slim.l2_regularizer(0.1))
        h = slim.fully_connected(h,1,activation_fn=None,weights_regularizer=slim.l2_regularizer(0.1))
    return h

def Whiv_ratio(M, n_layer=2, n_hidden=64, reuse=False):
    with tf.variable_scope("Whiv", reuse=reuse):
        h = slim.repeat(M,n_layer, slim.fully_connected,n_hidden,activation_fn=lrelu,weights_regularizer=slim.l2_regularizer(0.1))
        h = slim.fully_connected(h,1,activation_fn=None,weights_regularizer=slim.l2_regularizer(0.1))
    return h

