import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import pickle as pkl

import os
import sys
import argparse


""" tensorboard """
parser = argparse.ArgumentParser()
parser.add_argument('--logdir',type=str,default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),'tensorflow/mnist/logs/mnist_with_summaries'),help='Summaries log directory')
FLAGS, unparsed = parser.parse_known_args()
if tf.gfile.Exists(FLAGS.logdir):
    tf.gfile.DeleteRecursively(FLAGS.logdir)
tf.gfile.MakeDirs(FLAGS.logdir)


GPUID = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)
slim = tf.contrib.slim
ds = tf.contrib.distributions
st = tf.contrib.bayesflow.stochastic_tensor
graph_replace = tf.contrib.graph_editor.graph_replace

""" Parameters """
inp_data_dim = 10 #d
inp_cov_dim = 10 #d'
latent_dim = 300 #k
batch_size = 10 
eps_dim = 4
enc_net_hidden_dim = 128
n_samples = batch_size
n_epoch = 1
# filename = "M255.pkl"
""" Dataset """
def load_dataset():
    # raw_data=np.load('/opt/data/saket/gene_data/data/data_3k.npy')
    raw_data=np.load('gene_data/data/data_3k.npy')
    # cov = np.load('/opt/data/saket/gene_data/data/cov.npy')
    cov = np.load('gene_data/data/cov.npy')
    global inp_data_dim
    inp_data_dim = np.shape(raw_data)[1]
    global inp_cov_dim
    inp_cov_dim = np.shape(cov)[1]
    assert(np.shape(raw_data)[0] == np.shape(cov)[0])
    return raw_data[0:160], cov[0:160,:]

X_dataset, C_dataset = load_dataset()
XC_dataset = np.concatenate((X_dataset, C_dataset), axis=1)
print("Dataset Loaded... X:", np.shape(X_dataset), " C:", np.shape(C_dataset))

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
""" Networks """
def standard_normal(shape, **kwargs):
    """Create a standard Normal StochasticTensor."""
    return tf.cast(st.StochasticTensor(
        ds.MultivariateNormalDiag(tf.zeros(shape), tf.ones(shape), **kwargs)),  tf.float32)

def decoder_network(z1, z2, c, reuse):
    """ Create the decoder network with skip connections. 
    Arguments:
        z1: components of the latent features z which is given as input at layer 1
        z2: components of the latent features z which is given as input as layer 2
        c: covariates msatrix

    Return:
        y1, y2: output of the layer 1 and 2
    """
    assert(z1.get_shape().as_list()[0] == z2.get_shape().as_list()[0])
    with tf.variable_scope("decoder", reuse = reuse):
        A = tf.get_variable("A", shape=(inp_data_dim, latent_dim), initializer=tf.truncated_normal_initializer,regularizer=slim.l1_regularizer(scale=0.1))
        B = tf.get_variable("B", shape=(inp_data_dim, inp_cov_dim), initializer=tf.truncated_normal_initializer,regularizer=slim.l1_regularizer(scale=0.1))
        DELTA = tf.get_variable("DELTA", shape=(inp_data_dim), initializer=tf.truncated_normal_initializer) # It is a diagonal matrix
        DELTA = tf.diag(DELTA)
        
        y2 = tf.matmul(z2, A, transpose_b=True) + tf.matmul(c, B, transpose_b=True)
        y1 = y2 + tf.matmul(z1, DELTA, transpose_b=True)

        variable_summaries(A)
        variable_summaries(B)
        variable_summaries(DELTA)
        variable_summaries(y1)
        variable_summaries(y2)
        
    return y1, y2

def encoder_network(x, c, latent_dim, n_layer, z1_dim, z2_dim, eps_dim, reuse):
    with tf.variable_scope("encoder", reuse = reuse):
        eps2 = standard_normal([batch_size, eps_dim], name="eps2") * 1.0 # (batch_size, eps_dim)
        eps1 = standard_normal([batch_size, eps_dim], name="eps1") * 1.0 # (batch_size, eps_dim)

        h = tf.concat([x, c, eps1], 1)
        h = slim.repeat(h, n_layer, slim.fully_connected, latent_dim)
        z1 = slim.fully_connected(h, z1_dim)

        h = tf.concat([x, c, eps2, z1], axis=1)
        h = slim.repeat(h, n_layer, slim.fully_connected, latent_dim)
        z2 = slim.fully_connected(h, z2_dim)

        variable_summaries(z1)
        variable_summaries(z2)
    return z1, z2

def data_network(x, z, n_layer=3, n_hidden=256, reuse=False):
    """ The network to approximate the function g_si(x,z) whose optimal value will give w(x,z)
    Arguments:
        x: Data matrix of dimension (batch_size, inp_data_dim)
        z: Latent features of dimension (batch_size, z1+z2)
    Return:
        Evaluation of g_si(x,z) which is a scalar
        """
    with tf.variable_scope("data_network", reuse = reuse):
        h = tf.concat([x,z], axis=1)
        variable_summaries(h)
        h = slim.repeat(h, n_layer, slim.fully_connected, n_hidden, activation_fn=tf.tanh)
        variable_summaries(h)
        h = slim.fully_connected(h, 1, activation_fn=tf.tanh)
        variable_summaries(h)
        h = tf.Print(h,[h],message="h data_network")
        variable_summaries(h)
    return h

def cal_maximising_quantity(z_sample, x_sample, z, x, reuse):
    """ Expression which needs to be maximised for Si """
    #with tf.variable_scope("Si", reuse=reuse):
    # x_sample = tf.Print(x_sample, [x_sample], message="x_sample cal_mq")
    # z_sample = tf.Print(z_sample, [z_sample], message="z_sample cal_mq")
    # x = tf.Print(x, [x], message="x cal_mq")
    # z = tf.Print(z, [z], message="z cal_mq")
    variable_summaries(x_sample)
    variable_summaries(z_sample)
    variable_summaries(z)
    variable_summaries(x)
    g_si = data_network(x_sample, z_sample, reuse=False)
    f = -tf.nn.softplus(g_si, name="first_term")
    assert(f.get_shape() == (x_sample.get_shape().as_list()[0], 1))
    first_term = tf.truediv(tf.reduce_sum(f), f.get_shape().as_list()[0]*1.0)

    variable_summaries(g_si)
    variable_summaries(f)

    # g_si = data_network(x,z, reuse=True)
    g_si = graph_replace(g_si, {x_sample:x, z_sample:z})
    f = g_si - tf.nn.softplus(g_si)
    second_term = tf.truediv(tf.reduce_sum(f), f.get_shape().as_list()[0]*1.0)
    variable_summaries(g_si)
    variable_summaries(f)

    final = first_term+second_term
    tf.summary.scalar('first_term', first_term)
    tf.summary.scalar('second_term', second_term)
    tf.summary.scalar('final', final)
    return final, g_si

def cal_loss(x_sample, z_sample, x, z, reuse):
    #with tf.variable_scope("loss", reuse=reuse):
    # x_sample = tf.Print(x_sample, [x_sample], message="x_sample cal_loss")
    # z_sample = tf.Print(z_sample, [z_sample], message="z_sample cal_loss")
    # x = tf.Print(x, [x], message="x cal_loss")
    # z = tf.Print(z, [z], message="z cal_loss")
    w_x_z = data_network(x,z, reuse=True)
    f1 = tf.truediv(tf.reduce_sum(w_x_z), w_x_z.get_shape().as_list()[0]*1.0)
    f1 = tf.Print(f1,[w_x_z,f1], message="f1 loss")
    variable_summaries(x_sample)
    variable_summaries(z_sample)
    variable_summaries(z)
    variable_summaries(x)
    variable_summaries(w_x_z)
    w_x_z = data_network(x_sample, z_sample, reuse=True)
    f2 = tf.truediv(tf.reduce_sum(w_x_z), w_x_z.get_shape().as_list()[0]*1.0)
    f2 = tf.Print(f2,[w_x_z,f1], message="f2 loss")
    variable_summaries(w_x_z)
    final = f1-f2
    tf.summary.scalar('f1', f1)
    tf.summary.scalar('f2', f2)
    tf.summary.scalar('final', final)
    return final

""" Construct model """
tf.reset_default_graph()

x = tf.placeholder(tf.float32, shape=(batch_size, inp_data_dim))
c = tf.placeholder(tf.float32, shape=(batch_size, inp_cov_dim))

z1, z2 = encoder_network(x, c, enc_net_hidden_dim, 2, inp_data_dim, latent_dim, eps_dim, False)
y1, y2 = decoder_network(z1, z2, c, False)
z = tf.concat([z1,z2], axis=1)

MVN = ds.MultivariateNormalDiag(tf.zeros((latent_dim+inp_data_dim)), tf.ones((latent_dim+inp_data_dim)))
z_sample = MVN.sample(n_samples)
z1_sample = tf.slice(z_sample, [0, 0], [-1, inp_data_dim])
z2_sample = tf.slice(z_sample, [0, inp_data_dim], [-1, -1])
# x_sample, _ = decoder_network(z1_sample, z2_sample, c, True)
x_sample = graph_replace(y1, {z1:z1_sample, z2:z2_sample})

si_net_maximise, e1 = cal_maximising_quantity(z_sample, x_sample, z, x, False)
theta_phi_minimise = cal_loss(x_sample, z_sample, x, z, False, e1)
reconstruction_loss = tf.nn.l2_loss(y1-x)
tf.summary.scalar('si__net_maximise', si_net_maximise)
tf.summary.scalar('theta_phi_minimise', theta_phi_minimise)
variable_summaries(y1)
variable_summaries(x)
tf.summary.scalar('reconstruction_loss', reconstruction_loss)
t_vars = tf.trainable_variables()
print(t_vars)
svars = [var for var in t_vars if var.name.startswith("Si")]
dnvars = [var for var in t_vars if var.name.startswith("data_network")]
evars = [var for var in t_vars if var.name.startswith("encoder")]
dvars = [var for var in t_vars if var.name.startswith("decoder")]
lvars = [var for var in t_vars if var.name.startswith("loss")]

print("svars+dnvars",svars+dnvars)
print("e+d+l", evars+dvars+lvars)
opt = tf.train.AdamOptimizer(1e-4)
train_si = opt.minimize(-si_net_maximise, var_list=svars+dnvars)
reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
train_t_p = opt.minimize(theta_phi_minimise + sum(reg_variables), var_list=evars+dvars+lvars)

""" Training """
s_loss = []
tr_loss = []
recon_loss = []
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(FLAGS.logdir, graph = tf.get_default_graph())
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.2
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for epoch in range(n_epoch):
        np.random.shuffle(XC_dataset)
        X_dataset = XC_dataset[:,0:inp_data_dim]
        C_dataset = XC_dataset[:,inp_data_dim:]

        for i in range(np.shape(X_dataset)[0]//batch_size):
            xmb = X_dataset[i*batch_size:(i+1)*batch_size]
            cmb = C_dataset[i*batch_size:(i+1)*batch_size]
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            for _ in range(1):
                try:
                    summary, si_loss, _ = sess.run([merged, si_net_maximise, train_si], feed_dict={x:xmb, c:cmb}, options=run_options,run_metadata=run_metadata)
                except:
                    train_writer.close()
            for _ in range(1):
                try:
                    summary, t_loss, _ = sess.run([merged, theta_phi_minimise, train_t_p], feed_dict={x:xmb, c:cmb}, options=run_options,run_metadata=run_metadata)
                except:
                    train_writer.close()
            r_loss = sess.run(reconstruction_loss, feed_dict={x:xmb, c:cmb})
            s_loss.append(si_loss)
            tr_loss.append(t_loss)
            recon_loss.append(r_loss)

            try:
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
            except:
                train_writer.close()
        save_path = saver.save(sess, "gene_data/model.ckpt")
        print ("epoch:%d si_loss:%f t_loss:%f reconstruction_loss:%f"%(epoch, s_loss[-1], tr_loss[-1], recon_loss[-1]))

with open("/opt/data/saket/gene_data/s_loss.pkl", "wb") as pkl_file:
    pkl.dump(s_loss, pkl_file)
with open("/opt/data/saket/gene_data/tr_loss.pkl", "wb") as pkl_file:
    pkl.dump(tr_loss, pkl_file)
with open("/opt/data/saket/gene_data/recon_loss.pkl", "wb") as pkl_file:
    pkl.dump(recon_loss, pkl_file)
