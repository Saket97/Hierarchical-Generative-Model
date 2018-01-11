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
batch_size = 160 
eps_dim = 10
enc_net_hidden_dim = 256
n_samples = batch_size
n_epoch = 10000
# filename = "M255.pkl"
""" Dataset """
def load_dataset():
    raw_data=np.load('/opt/data/saket/gene_data/data/data_3k.npy')
    #raw_data=np.load('gene_data/data/data_3k.npy')
    cov = np.load('/opt/data/saket/gene_data/data/cov1.npy')
    #cov = np.load('gene_data/data/cov1.npy')
    global inp_data_dim
    inp_data_dim = np.shape(raw_data)[1]
    global inp_cov_dim
    inp_cov_dim = np.shape(cov)[1]
    assert(np.shape(raw_data)[0] == np.shape(cov)[0])
    print("raw min",np.min(raw_data))
    print("raw max",np.max(raw_data))
    print("cov min",np.min(cov))
    print("cov max",np.max(cov))
    return raw_data[0:160], cov[0:160,:], raw_data[160:], cov[160:]

X_dataset, C_dataset, X_t, C_t = load_dataset()
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

def encoder_network(x, c, latent_dim, n_layer, z1_dim, z2_dim, eps_dim, reuse):
    with tf.variable_scope("encoder", reuse = reuse):
        eps2 = standard_normal([batch_size, eps_dim], name="eps2") * 1.0 # (batch_size, eps_dim)
        eps1 = standard_normal([batch_size, eps_dim], name="eps1") * 1.0 # (batch_size, eps_dim)

        h = tf.concat([x, c, eps1], 1)
        h = slim.repeat(h, n_layer, slim.fully_connected, latent_dim, activation_fn=tf.tanh, biases_initializer=tf.truncated_normal_initializer, weights_regularizer = slim.l2_regularizer(0.5), biases_regularizer=slim.l2_regularizer(0.5))
        z1 = slim.fully_connected(h, z1_dim, activation_fn=tf.tanh, biases_initializer=tf.truncated_normal_initializer, weights_regularizer = slim.l2_regularizer(0.5), biases_regularizer=slim.l2_regularizer(0.5))

        h = tf.concat([x, c, z1, eps2], axis=1)
        h = slim.repeat(h, n_layer, slim.fully_connected, latent_dim, activation_fn=tf.tanh, biases_initializer=tf.truncated_normal_initializer, weights_regularizer = slim.l2_regularizer(0.5), biases_regularizer=slim.l2_regularizer(0.5))
        z2 = slim.fully_connected(h, z2_dim, activation_fn=tf.tanh, biases_initializer=tf.truncated_normal_initializer, weights_regularizer = slim.l2_regularizer(0.5), biases_regularizer=slim.l2_regularizer(0.5))

        variable_summaries(z1)
        variable_summaries(z2)
    return z1, z2

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
        inp = tf.concat([z2,c], axis=1)
        y2 = tf.layers.dense(inp, inp_data_dim, use_bias=False, kernel_initializer=tf.truncated_normal_initializer, kernel_regularizer=slim.l1_regularizer(0.5))
        DELTA = tf.get_variable("DELTA", shape=(inp_data_dim), initializer=tf.truncated_normal_initializer) # It is a diagonal matrix
        y1 = y2+DELTA*z1
        
        variable_summaries(DELTA)
        variable_summaries(y1)
        variable_summaries(y2)
        
    return y1, y2

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
        h = slim.repeat(h, n_layer, slim.fully_connected, n_hidden, activation_fn=tf.tanh, biases_initializer=tf.truncated_normal_initializer, weights_regularizer = slim.l2_regularizer(0.5), biases_regularizer=slim.l2_regularizer(0.5))
        variable_summaries(h)
        h = slim.fully_connected(h, 1, activation_fn=tf.tanh, biases_initializer=tf.truncated_normal_initializer, weights_regularizer = slim.l2_regularizer(0.5), biases_regularizer=slim.l2_regularizer(0.5))
        variable_summaries(h)
        # h = tf.Print(h,[h],message="h data_network")
        # variable_summaries(h)
    return h

def cal_loss(x, z, x_sample, z_sample):
    with tf.variable_scope("Loss"):
        g_x_z = data_network(x, z)
        g_x_z_s = graph_replace(g_x_z, {x:x_sample, z:z_sample})
        with tf.name_scope("Si_maximise"):
            f1 = g_x_z - tf.nn.softplus(g_x_z)
            f2 = -tf.nn.softplus(g_x_z_s)
            f1 = tf.truediv(tf.reduce_sum(f1), f1.get_shape().as_list()[0]*1.0)
            f2 = tf.truediv(tf.reduce_sum(f2), f2.get_shape().as_list()[0]*1.0)
            si = f1+f2
            # si = f1
        
        with tf.name_scope("theta_phi"):
            f1 = tf.truediv(tf.reduce_sum(g_x_z), g_x_z.get_shape().as_list()[0]*1.0)
            f2 = tf.truediv(tf.reduce_sum(g_x_z_s), g_x_z_s.get_shape().as_list()[0]*1.0)
            tp = f1-f2
            # tp = f1
    return si, tp

def train(si, tp, x, c, recon_loss):

    t_vars = tf.trainable_variables()
    evars = [var for var in t_vars if var.name.startswith("encoder")]
    dvars = [var for var in t_vars if var.name.startswith("decoder")]
    lvars = [var for var in t_vars if var.name.startswith("Loss")]
    r_loss = tf.losses.get_total_loss() 
    opt = tf.train.AdamOptimizer(1e-4)
    train_si = opt.minimize(-si+r_loss, var_list=lvars)
    train_t_p = opt.minimize(tp+r_loss, var_list=evars+dvars)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.logdir, graph = tf.get_default_graph())
    s_loss = []
    tp_loss = []
    re_loss = []
    with tf.Session() as sess:
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
                    # try:
                    if epoch % 100 == 0 and i == 0:
                        summary, si_loss, _ = sess.run([merged, si, train_si], feed_dict={x:xmb, c:cmb}, options=run_options,run_metadata=run_metadata)
                        train_writer.add_run_metadata(run_metadata, 'step%03depoch%d' % (i, epoch))
                        train_writer.add_summary(summary, i)
                    else:                    
                        si_loss, _ = sess.run([si, train_si], feed_dict={x:xmb, c:cmb}, options=run_options,run_metadata=run_metadata)
                    s_loss.append(si_loss)
                    # except:
                        # train_writer.close()
                
                for _ in range(1):
                    #try:
                    if epoch % 100 == 0 and i == 0:
                        summary, t_loss, _ = sess.run([merged, tp, train_t_p], feed_dict={x:xmb, c:cmb}, options=run_options,run_metadata=run_metadata)
                        train_writer.add_run_metadata(run_metadata, 'step%03depochSecond%d' % (i, epoch))
                        train_writer.add_summary(summary, i)
                    else:
                        t_loss, _ = sess.run([tp, train_t_p], feed_dict={x:xmb, c:cmb}, options=run_options,run_metadata=run_metadata)
                    tp_loss.append(t_loss)
                    #except:
                rec_loss = sess.run(recon_loss, feed_dict={x:xmb,c:cmb})
                re_loss.append(rec_loss)                
                #train_writer.close()

                print ("sTEP:%d si_loss:%f t_loss:%f recon_loss:%f"%(i, s_loss[-1], tp_loss[-1], re_loss[-1]))
                
            save_path = saver.save(sess, "gene_data/model.ckpt")
            print ("######################## epoch:%d si_loss:%f t_loss:%f"%(epoch, s_loss[-1], tp_loss[-1]))
    train_writer.close()
    with open("gene_data/data/s_loss.pkl", "wb") as pkl_file:
        pkl.dump(s_loss, pkl_file)
    with open("gene_data/data/tp_loss.pkl", "wb") as pkl_file:
        pkl.dump(tp_loss, pkl_file)
    with open("gene_data/data/re_loss.pkl", "wb") as pkl_file:
        pkl.dump(re_loss, pkl_file)

def main():
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
    x_sample = graph_replace(y1, {z1:z1_sample, z2:z2_sample})

    si, tp = cal_loss(x, z, x_sample, z_sample)
    recon_loss = tf.nn.l2_loss(y1-x)
    train(si, tp, x, c,recon_loss)

main()
