import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import pickle as pkl

slim = tf.contrib.slim
ds = tf.contrib.distributions
st = tf.contrib.bayesflow.stochastic_tensor
graph_replace = tf.contrib.graph_editor.graph_replace

""" Parameters """
inp_data_dim = 40 #d
latent_dim = 25 #k
batch_size = 100
eps_dim = 10
enc_net_hidden_dim = 256
n_samples = 100
n_epoch = 1
filename = "M255.pkl"
""" Dataset """
def load_dataset():
    with open(filename, "rb") as pkl_file:
        a = pkl.load(pkl_file)
    # a = a[:][0:40]
    a = np.array(a)
    a = a[:,0:40]
    return a
X_dataset = load_dataset()
print("X_dataset:", np.shape(X_dataset))

""" Networks """
def standard_normal(shape, **kwargs):
    """Create a standard Normal StochasticTensor."""
    return tf.cast(st.StochasticTensor(
        ds.MultivariateNormalDiag(tf.zeros(shape), tf.ones(shape), **kwargs)),  tf.float32)

def decoder_network(z1, z2):
    """ Create the decoder network with skip connections. 
    Arguments:
        z1: components of the latent features z which is given as input at layer 1
        z2: components of the latent features z which is given as input as layer 2

    Return:
        y1, y2: output of the layer 1 and 2
    """
    assert(z1.get_shape().as_list()[0] == z2.get_shape().as_list()[0])
    with tf.variable_scope("decoder", reuse = tf.AUTO_REUSE):
        A = tf.get_variable("A", shape=(inp_data_dim, latent_dim), regularizer=slim.l1_regularizer(scale=0.1))
        DELTA = tf.get_variable("DELTA", shape=(inp_data_dim)) # It is a diagonal matrix
        DELTA = tf.diag(DELTA)
        
        y2 = tf.matmul(z2, A, transpose_b=True)
        y1 = y2 + tf.matmul(z1, DELTA, transpose_b=True)
    return y1, y2

def encoder_network(x, latent_dim, n_layer, z1_dim, z2_dim, eps_dim):
    with tf.variable_scope("encoder", reuse = tf.AUTO_REUSE):
        eps1 = standard_normal([batch_size, eps_dim], name="eps1") * 1.0 # (batch_size, eps_dim)
        eps2 = standard_normal([batch_size, eps_dim], name="eps2") * 1.0 # (batch_size, eps_dim)

        h = tf.concat([x, eps1], 1)
        h = slim.repeat(h, n_layer, slim.fully_connected, latent_dim, activation_fn=tf.nn.relu)
        z1 = slim.fully_connected(h, z1_dim, activation_fn=tf.nn.relu)

        h = tf.concat([x, eps2, z1], axis=1)
        h = slim.repeat(h, n_layer, slim.fully_connected, latent_dim, activation_fn=tf.nn.relu)
        z2 = slim.fully_connected(h, z2_dim, activation_fn=tf.nn.relu)
    return z1, z2

def data_network(x, z, n_layer=2, n_hidden=256):
    """ The network to approximate the function g_si(x,z) whose optimal value will give w(x,z)
    Arguments:
        x: Data matrix of dimension (batch_size, inp_data_dim)
        z: Latent features of dimension (batch_size, z1+z2)
    Return:
        Evaluation of g_si(x,z) which is a scalar
        """
    with tf.variable_scope("data_network", reuse = tf.AUTO_REUSE):
        h = tf.concat([x,z], axis=1)
        h = slim.repeat(h, n_layer, slim.fully_connected, n_hidden, activation_fn=tf.nn.relu)
        h = slim.fully_connected(h, 1, activation_fn=tf.nn.relu)
    # print("data network shape ", h.get_shape().as_list())
    return h

def cal_maximising_quantity(z_sample, x_sample, z, x):
    """ Expression which needs to be maximised for Si """
    with tf.variable_scope("Si", reuse=tf.AUTO_REUSE):
        g_si = data_network(x_sample, z_sample)
        # g_si = tf.Print(g_si, [g_si], message="g_si for first term in Si")
        g_si_sig = tf.nn.sigmoid(g_si)
        # g_si_sig = tf.Print(g_si_sig, [g_si_sig], message="g_s_sig for first term")
        f = tf.log(1-g_si_sig)
        # f = tf.Print(f, [f], message="first_term Si")
        assert(f.get_shape() == (x_sample.get_shape().as_list()[0], 1))
        first_term = tf.truediv(tf.reduce_sum(f), f.get_shape().as_list()[0]*1.0)

        g_si = data_network(x,z)
        # g_si = tf.Print(g_si, [g_si], message="g_si for second term in Si")
        g_si_sig = tf.nn.sigmoid(g_si)
        # g_si_sig = tf.Print(g_si_sig, [g_si_sig], message="g_si_sig for second term in Si")
        f = tf.log(g_si_sig)
        # f = tf.Print(f, [f], message="Second_term Si")
        second_term = tf.truediv(tf.reduce_sum(f), f.get_shape().as_list()[0]*1.0)

        final = first_term+second_term
    return final

def cal_loss(x_sample, z_sample, x, z):
    with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
        w_x_z = data_network(x,z)
        f1 = tf.truediv(tf.reduce_sum(w_x_z), w_x_z.get_shape().as_list()[0]*1.0)

        w_x_z = data_network(x_sample, z_sample)
        f2 = tf.truediv(tf.reduce_sum(w_x_z), w_x_z.get_shape().as_list()[0]*1.0)

        final = f1-f2
    return final

""" Construct model """
tf.reset_default_graph()

x = tf.placeholder(tf.float32, shape=(None, inp_data_dim))

z1, z2 = encoder_network(x, enc_net_hidden_dim, 2, inp_data_dim, latent_dim, eps_dim)
y1, y2 = decoder_network(z1, z2)
z = tf.concat([z1,z2], axis=1)

# MVN = ds.MultivariateNormalDiag(tf.zeros((batch_size, latent_dim+inp_data_dim)), tf.ones((batch_size, latent_dim+inp_data_dim)))
MVN = ds.MultivariateNormalDiag(tf.zeros((latent_dim+inp_data_dim), dtype = tf.float32)*1.0, tf.ones((latent_dim+inp_data_dim), dtype=tf.float32), allow_nan_stats=False)
# MVN.dtype = tf.float32
z_sample = MVN.sample(n_samples)
z1_sample = tf.slice(z_sample, [0, 0], [-1, inp_data_dim])
z2_sample = tf.slice(z_sample, [0, inp_data_dim], [-1, -1])
# z1_sample = tf.Print(z1_sample, [z1_sample, z2_sample], message="z1_sample, z2_sampple")
x_sample, _ = decoder_network(z1_sample, z2_sample)

si_net_maximise = cal_maximising_quantity(z_sample, x_sample, z, x)
theta_phi_minimise = cal_loss(x_sample, z_sample, x, z)
reconstruction_loss = tf.nn.l2_loss(y1-x)

t_vars = tf.trainable_variables()
svars = [var for var in t_vars if var.name.startswith("Si")]
dnvars = [var for var in t_vars if var.name.startswith("data_network")]
evars = [var for var in t_vars if var.name.startswith("encoder")]
dvars = [var for var in t_vars if var.name.startswith("decoder")]
lvars = [var for var in t_vars if var.name.startswith("loss")]

opt = tf.train.AdamOptimizer(1e-3, beta1=0.5)
train_si = opt.minimize(-si_net_maximise, var_list=svars+dnvars)
reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
train_t_p = opt.minimize(theta_phi_minimise+sum(reg_variables), var_list=evars+dvars+lvars)

""" Training """
s_loss = []
tr_loss = []
recon_loss = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epoch):
        np.random.shuffle(X_dataset)

        for i in range(10):
            xmb = X_dataset[i*batch_size:(i+1)*batch_size]

            for _ in range(1):
                si_loss, _ = sess.run([si_net_maximise, train_si], feed_dict={x:xmb})
            for _ in range(1):
                t_loss, _ = sess.run([theta_phi_minimise, train_t_p], feed_dict={x:xmb})
            r_loss = sess.run(reconstruction_loss, feed_dict={x:xmb})
            s_loss.append(si_loss)
            tr_loss.append(t_loss)
            recon_loss.append(r_loss)
        print ("epoch:%d si_loss:%f t_loss:%f reconstruction_loss:%f"%(epoch, s_loss[-1], tr_loss[-1], recon_loss[-1]))

with open("s_loss.pkl", "wb") as pkl_file:
    pkl.dump(s_loss, pkl_file)
with open("tr_loss.pkl", "wb") as pkl_file:
    pkl.dump(tr_loss, pkl_file)
with open("recon_loss.pkl", "wb") as pkl_file:
    pkl.dump(recon_loss, pkl_file)

# Not for use in this file
# def normalize_each_class(X, y, c):
#     i2 = (y==2).nonzero()[0]
#     i1 = (y==1).nonzero()[0]
#     i0 = (y==0).nonzero()[0]
#     data2 = X[i2]
#     data1 = X[i1]
#     data0 = X[i0]
#     m = np.mean(data2, axis=0)
#     print ("Mean for second class:",m)
#     data2 = data2-1.0*m
#     maximum = 1e-6 + np.amax(np.abs(data2), axis=0)
#     data2 /= maximum
#     m = np.mean(data1, axis=0)
#     print ("Mean for first class:",m)
#     data1 = data1-1.0*m
#     maximum = 1e-6 + np.amax(np.abs(data1), axis=0)
#     data1 /= maximum
#     m = np.mean(data0, axis=0)
#     print ("Mean for zeroth class:",m)
#     data0 = data0-1.0*m
#     maximum = 1e-6 + np.amax(np.abs(data0), axis=0)
#     data0 /= maximum
#     data = np.concatenate((data2, data1, data0), axis=0)
#     s = data.shape[1]
#     labels = np.concatenate((y[i2],y[i1],y[i0]))
#     c = np.concatenate((c[i2],c[i1],c[i0]), axis=0)
#     sc = c.shape[1]
#     dl = np.concatenate((data,c,np.expand_dims(labels, 1)), axis=1)
#     np.random.shuffle(dl)
#     data = dl[:,0:s]
#     cov = dl[:,s:s+sc]
#     labels = dl[:,s+sc:]
#     assert(labels.shape[1]==1)
#     return data, np.squeeze(labels), cov

# def load_minibatch():
#     i2 = (raw_labels==2).nonzero()[0]
#     i1 = (raw_labels==1).nonzero()[0]
#     i0 = (raw_labels==0).nonzero()[0]
#     n2 = batch_size//3
#     n1 = n2
#     n0 = batch_size-n1-n2
#     np.random.shuffle(i2)
#     np.random.shuffle(i1)
#     np.random.shuffle(i0)
#     i =  np.concatenate((i2[0:n2],i1[0:n1],i0[0:n0]))
#     np.random.shuffle(i)
#     return i
# def get_indices():
#     i2 = (raw_labels==2).nonzero()[0]
#     i1 = (raw_labels==1).nonzero()[0]
#     i0 = (raw_labels==0).nonzero()[0]
#     a = np.amin([i2.shape[0],i1.shape[0],i0.shape[0]])
#     np.random.shuffle(i2) 
#     np.random.shuffle(i1) 
#     np.random.shuffle(i0)
#     r = np.concatenate((i2[0:a],i1[0:a], i0[0:a]))
#     np.random.shuffle(r)
    # return r 