import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import pickle as pkl
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np

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
num_classes = 2
inp_data_dim = 10 #d
inp_cov_dim = 10 #d'
latent_dim = 40 #k
batch_size = 300 
test_batch_size = 500
eps2_dim = 20
eps1_dim = 10
enc_net_hidden_dim = 512
n_samples = batch_size
n_epoch = 500
niter_clf = 15000
cyc_x = 5
cyc_z = 2
cyc1 = 0
include_cyc = True
keep_prob = 1
n_train = 1500
n_test = 500
# filename = "M255.pkl"
""" Dataset """
def load_dataset():
    raw_data=np.load('/opt/data/saket/gene_data/data/toy/mod_total_data.npy')
    #raw_data=np.load('gene_data/data/data_3k.npy')
    cov = np.load('/opt/data/saket/gene_data/data/toy/cov.npy')
    print ("cov:", cov)
    labels = np.load('/opt/data/saket/gene_data/data/toy/data_label.npy')
    #cov = np.load('gene_data/data/cov1.npy')
    #raw_data = np.log10(raw_data+0.1)
    cov = np.log10(cov+0.1)
    global inp_data_dim
    inp_data_dim = np.shape(raw_data)[1]
    global inp_cov_dim
    inp_cov_dim = np.shape(cov)[1]
    print ("inp_cov_dim:", inp_cov_dim)
    assert(np.shape(raw_data)[0] == np.shape(cov)[0])
    raw_data_mean = np.mean(raw_data, axis=0)
    raw_data = (raw_data-1.0*raw_data_mean)
    print("raw min",np.min(raw_data))
    cov_data_mean = np.mean(cov, axis=0)
    #cov_data = (cov-1.0*cov_data_mean)
    print("raw max",np.max(raw_data))
    print("cov min",np.min(cov))
    print("cov max",np.max(cov))
    return raw_data[0:n_train], cov[0:n_train], labels[0:n_train], raw_data[n_train:n_train+n_test], cov[n_train:n_train+n_test], labels[n_train:n_train+n_test]

X_dataset, C_dataset, raw_labels, X_t, C_t, test_labels = load_dataset()
XC_dataset = np.concatenate((X_dataset, C_dataset), axis=1)
print("Dataset Loaded... X:", np.shape(X_dataset), " C:", np.shape(C_dataset))

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
""" Networks """
def standard_normal(shape, **kwargs):
    """Create a standard Normal StochasticTensor."""
    return tf.cast(st.StochasticTensor(
        ds.MultivariateNormalDiag(tf.zeros(shape), tf.ones(shape), **kwargs)),  tf.float32)

def encoder_network(x, c, latent_dim, n_layer, z1_dim, z2_dim, eps1, eps2, reuse, prob):
    with tf.variable_scope("encoder", reuse = reuse):

        h = tf.concat([x, c, eps1], 1)
        h = slim.repeat(h, n_layer, slim.fully_connected, latent_dim, activation_fn=tf.tanh, biases_initializer=tf.truncated_normal_initializer(), weights_regularizer = slim.l2_regularizer(0.1))
        variable_summaries(h, name="enc_z1_hidden_layer_output")
        z1 = slim.fully_connected(h, z1_dim, activation_fn=tf.tanh, biases_initializer=tf.truncated_normal_initializer, weights_regularizer = slim.l2_regularizer(0.1))
        z1 = tf.nn.dropout(z1, prob)

        h = tf.concat([x, c, z1, eps2], axis=1)
        h = slim.repeat(h, n_layer, slim.fully_connected, latent_dim, activation_fn=tf.nn.elu, biases_initializer=tf.truncated_normal_initializer, weights_regularizer = slim.l2_regularizer(0.1))
        variable_summaries(h, name="enc_z2_hidden_layer_output")
        z2 = slim.fully_connected(h, z2_dim, activation_fn=None, biases_initializer=tf.truncated_normal_initializer, weights_regularizer = slim.l2_regularizer(0.1))
        
        #z1 = tf.Print(z1, [z1], message="z1")
        #z2 = tf.Print(z2, [z2], message="z2")
        variable_summaries(z1, name="z1")
        variable_summaries(z2, name="z2")
    return z1, z2

def decoder_network(z1, z2, c, reuse, noise):
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
        y2 = tf.layers.dense(inp, inp_data_dim, use_bias=False, kernel_initializer=tf.orthogonal_initializer(gain=3), kernel_regularizer=slim.l1_regularizer(1.0))
        print("y2_name:",y2.name)
        weights = tf.get_default_graph().get_tensor_by_name("decoder/dense"+"/kernel:0")
        DELTA = tf.get_variable("DELTA", shape=(inp_data_dim), initializer=tf.truncated_normal_initializer) # It is a diagonal matrix
        y1 = y2+DELTA*z1

        A = tf.slice(weights, [0,0],[latent_dim,-1])
        A = tf.transpose(A)
        B = tf.slice(weights, [latent_dim, 0], [-1,-1])
        B = tf.transpose(B)
        variable_summaries(DELTA, name="DELTA")
        variable_summaries(y1, name="y1")
        variable_summaries(y2, name="y2")
        variable_summaries(A, name="A")
        variable_summaries(B, name="B")
        #y1 = tf.Print(y1, [y1], message="y1")        
    return y1, y2, A, B, DELTA

def data_network(x, z, n_layer=2, n_hidden=1024, reuse=False):
    """ The network to approximate the function g_si(x,z) whose optimal value will give w(x,z)
    Arguments:
        x: Data matrix of dimension (batch_size, inp_data_dim)
        z: Latent features of dimension (batch_size, z1+z2)
    Return:
        Evaluation of g_si(x,z) which is a scalar
        """
    #x = tf.Print(x, [x], message="x data network")
    #z = tf.Print(z, [z], message="z data network")
    with tf.variable_scope("data_network", reuse = reuse):
        h = tf.concat([x,z], axis=1)
        #variable_summaries(h)
        h = slim.repeat(h, n_layer, slim.fully_connected, n_hidden, activation_fn=tf.nn.elu, biases_initializer=tf.truncated_normal_initializer, weights_regularizer = slim.l2_regularizer(0.1))
        variable_summaries(h, name="data_network_hidden")
        h = slim.fully_connected(h, 1, activation_fn=None, biases_initializer=tf.truncated_normal_initializer, weights_regularizer = slim.l2_regularizer(0.1))
        variable_summaries(h, name="data_net_output")
        #h = tf.Print(h,[h],message="h data_network")
        # variable_summaries(h)
    return h

def transform_z2(z2, reuse=False):
    with tf.variable_scope("transform_z2", reuse = reuse):
        h = slim.repeat(z2, 3, slim.fully_connected, 32, activation_fn=tf.nn.elu, weights_regularizer = slim.l2_regularizer(0.1))
        h = slim.fully_connected(h,latent_dim, activation_fn=None, weights_regularizer=slim.l2_regularizer(0.1))
    return h

def cal_loss(x, z_list, y1_list, x_sample, z_sample, z_x_sample_encoded_list):
    with tf.variable_scope("Loss"):
        g_x_z1 = data_network(x, z_list[0])
        g_x_z_list = [g_x_z1]
        g_x_z_log_list = [g_x_z1 - tf.nn.softplus(g_x_z1)]
        for i in range(1,len(z_list)):
            g_x_z1 = data_network(x,z_list[i],reuse=True)
            # g_x_z1 = graph_replace(g_x_z1, {z_list[i-1]:z_list[i]})
            g_x_z_list.append(g_x_z1)
            g_x_z_log_list.append(g_x_z1 - tf.nn.softplus(g_x_z1))
        g_x_z = -tf.truediv(tf.add_n(g_x_z_list),len(g_x_z_list)*1.0)

        g_x_z_s = data_network(x_sample, z_sample, reuse=True)
        # g_x_z_s = graph_replace(g_x_z1, {x:x_sample, z_list[0]:z_sample})
        
        with tf.name_scope("Si_maximise"):
            tmp = tf.add_n(g_x_z_log_list)/(len(g_x_z_log_list)*1.0)
            f1 = tf.reduce_sum(tmp)/(tmp.get_shape().as_list()[0]*1.0)
            f2 = -tf.nn.softplus(g_x_z_s)
            f2 = tf.reduce_sum(f2)/(f2.get_shape().as_list()[0]*1.0)
            si = f1+f2 # maximize this quantity

        with tf.name_scope("phi"):
            g_x_z = tf.add_n(g_x_z_list)/(len(g_x_z_list)*1.0)
            p = tf.reduce_sum(g_x_z)/(g_x_z.get_shape().as_list()[0]*1.0)
            # Adding cyclic consistency
            if include_cyc:
                norm_list = []
                norm_inv_list = []
                for i in range(len(y1_list)):
                    norm = tf.norm(x-y1_list[i], axis=1)
                    norm_list.append(norm)
                    norm_inv_list.append(tf.norm(y1_list[i], axis=1))
                norm = tf.add_n(norm_list)/(len(norm_list)*1.0)
                norm_inv = tf.add_n(norm_inv_list)/(len(norm_inv_list)*1.0)
                norm = tf.reduce_sum(norm)/(norm.get_shape().as_list()[0]*1.0)
                norm_inv = tf.reduce_sum(norm_inv)/(norm_inv.get_shape().as_list()[0]*1.0)
                p = p+cyc_x*norm+cyc1/norm_inv

        with tf.name_scope("theta"):
            t = tf.reduce_sum(g_x_z_s)/(g_x_z_s.get_shape().as_list()[0]*1.0)
            t = -t
            if include_cyc:
                norm_list = []
                for i in range(len(z_x_sample_encoded_list)):
                    norm = tf.norm(z_sample-z_x_sample_encoded_list[i], axis=1)
                    norm_list.append(norm)
                norm = tf.add_n(norm_list)/(len(norm_list)*1.0)
                norm = tf.reduce_sum(norm)/(norm.get_shape().as_list()[0]*1.0)
                t = t+cyc_z*norm
            
    return si, t, p

def classifier(z_input, labels, reuse):
    with tf.variable_scope("Classifier", reuse = reuse):
        out_logits = tf.layers.dense(z_input, num_classes, use_bias=False, kernel_initializer=tf.truncated_normal_initializer, kernel_regularizer=slim.l2_regularizer(0.1))
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out_logits, labels=labels)
        #variable_summaries(loss, name="softmax_loss")
        loss = tf.Print(loss, [loss], message="softmax loss")
        loss = tf.reduce_mean(loss)
        weights = tf.get_default_graph().get_tensor_by_name("Classifier/dense"+"/kernel:0")
        #variable_summaries(weights, name="Classifier_weights")
       # loss = tf.Print(loss, [loss], message="loss mean")
        return loss, out_logits, tf.nn.softmax(out_logits)

def get_indices():
    i2 = (raw_labels==2).nonzero()[0]
    i1 = (raw_labels==1).nonzero()[0]
    i0 = (raw_labels==0).nonzero()[0]
    a = np.amin([i2.shape[0],i1.shape[0],i0.shape[0]])
    np.random.shuffle(i2) 
    np.random.shuffle(i1) 
    np.random.shuffle(i0)
    r = np.concatenate((i2[0:a],i1[0:a], i0[0:a]))
    np.random.shuffle(r)
    return r 

def train(si, t, p, x, c, recon_loss, y1, z1, z2, recon_abs, recon_std, A, B, D, z_assign, z_input, labels, loss, out_logits, out_soft, prob):

    t_vars = tf.trainable_variables()
    evars = [var for var in t_vars if var.name.startswith("encoder")]
    dvars = [var for var in t_vars if var.name.startswith("decoder")]
    lvars = [var for var in t_vars if var.name.startswith("Loss")]
    cvars = [var for var in t_vars if var.name.startswith("Classifier")]
    tz2vars = [var for var in t_vars if var.name.startswith("transform_z2")]
    #assert(len(t_vars) == len(evars)+len(dvars)+len(lvars)+len(cvars)+len(tz2vars))
    print("len(t_vars)",len(t_vars))
    print("len(evars)+len(dvars)+len(lvars)+len(cvars)+len(tz2vars)",len(evars)+len(dvars)+len(lvars)+len(cvars)+len(tz2vars))
    print("tvars:",t_vars)
    print("evars:", evars)
    print("dvars:",dvars)
    print("lvars:",lvars)
    print("cvars:",cvars)
    print("tz2vars:",tz2vars)
    r_loss = tf.losses.get_regularization_loss() 
    r_loss_clf = tf.losses.get_regularization_loss(scope="Classifier")
    r_loss = r_loss - r_loss_clf
    opt = tf.train.AdamOptimizer(1e-4)
    train_si = opt.minimize(-si+r_loss, var_list=lvars)
    if include_cyc:
        train_t = opt.minimize(t+r_loss, var_list=dvars+tz2vars)
        train_p = opt.minimize(p+r_loss, var_list=evars+tz2vars)
    else:
        tp = t+p+r_loss
        train_tp = opt.minimize(t+p+r_loss, var_list = dvars+evars+tz2vars)

    train_clf = opt.minimize(loss+r_loss_clf, var_list=cvars)

    merged = tf.summary.merge_all()
    s_loss = []
    t_loss_list = []
    p_loss_list = []
    re_loss = []
    re_abs = []
    re_std = []
    acc = []
    acc_abs = []
    acc_std = []
    train_writer = tf.summary.FileWriter(FLAGS.logdir, graph = tf.get_default_graph())
    config = projector.ProjectorConfig()
    
    embedding = config.embeddings.add()
    embedding_var = tf.Variable(initial_value = np.zeros((z1.shape), dtype=np.float32), name="z1_embed_var")
    assign_z1 = tf.assign(embedding_var, z1)
    embedding.tensor_name = embedding_var.name
    projector.visualize_embeddings(train_writer, config)        
   
    embedding_var = tf.Variable(initial_value = np.zeros((z2.shape), dtype=np.float32), name="z2_embed_var") 
    assign_z2 = tf.assign(embedding_var, z2)
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    projector.visualize_embeddings(train_writer, config)        
    
   # embedding = config.embeddings.add()
   # embedding.tensor_name = x.name 
   # projector.visualize_embeddings(train_writer, config)        
    #weights = tf.get_variable("decoder/dense/kernel:0")
    #print(tf.trainable_variables())
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer(), feed_dict={x:XC_dataset[0:batch_size,0:inp_data_dim], c:XC_dataset[0:batch_size,inp_data_dim:]})
        saver = tf.train.Saver(save_relative_paths=True)
        # sess.run(z_assign, feed_dict={x:XC_dataset[:,0:inp_data_dim], c:XC_dataset[:,inp_data_dim:]})
        #saver.restore(sess, os.path.join("/opt/data/saket/model_2gg00_100_5/", "model.ckpt-4000"))
        for epoch in range(n_epoch):
            #np.random.shuffle(XC_dataset)
            X_dataset = XC_dataset[:,0:inp_data_dim]
            C_dataset = XC_dataset[:,inp_data_dim:]
            sess.run(z_assign, feed_dict={x:XC_dataset[0:batch_size,0:inp_data_dim], c:XC_dataset[0:batch_size,inp_data_dim:], prob:keep_prob})

            for i in range(np.shape(X_dataset)[0]//batch_size):
                xmb = X_dataset[i*batch_size:(i+1)*batch_size]
                cmb = C_dataset[i*batch_size:(i+1)*batch_size]
                for j in range(1):
                    # try:
                    if epoch % 100 == 0 and i == 0:
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        summary, si_loss, _ , _ , _ = sess.run([merged, si, train_si, assign_z1, assign_z2], feed_dict={x:xmb, c:cmb, prob:keep_prob}, options=run_options,run_metadata=run_metadata)
                        train_writer.add_run_metadata(run_metadata, 'step%d' % (epoch*6+j))
                        train_writer.add_summary(summary, epoch*6+j)
                    else:                    
                        si_loss, _ , _ , _ = sess.run([si, train_si, assign_z1, assign_z2], feed_dict={x:xmb, c:cmb, prob:keep_prob})
                    s_loss.append(si_loss)
                    # except:
                        # train_writer.close()
                
                for j in range(1):
                    #try:
                    if epoch % 100 == 0 and i == 0:
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        if include_cyc:
                            summary, t_loss, _ , _ , _  = sess.run([merged, t, train_t, assign_z1, assign_z2], feed_dict={x:xmb, c:cmb, prob:keep_prob}, options=run_options,run_metadata=run_metadata)
                            summary, p_loss, _ , _ , _  = sess.run([merged, p, train_p, assign_z1, assign_z2], feed_dict={x:xmb, c:cmb, prob:keep_prob}, options=run_options,run_metadata=run_metadata)
                        else:
                            summary, tp_loss, _ , _ , _  = sess.run([merged, tp, train_tp, assign_z1, assign_z2], feed_dict={x:xmb, c:cmb, prob:keep_prob}, options=run_options,run_metadata=run_metadata)
                        train_writer.add_run_metadata(run_metadata, 'step%d' % (epoch*6+5+j))
                        train_writer.add_summary(summary, epoch*6+5+j)
                    else:
                        if include_cyc:
                            t_loss, _ , _ , _ = sess.run([t, train_t, assign_z1, assign_z2], feed_dict={x:xmb, c:cmb, prob:keep_prob})
                            p_loss, _ , _ , _ = sess.run([p, train_p, assign_z1, assign_z2], feed_dict={x:xmb, c:cmb, prob:keep_prob})
                        else:
                            summary, tp_loss, _ , _ , _  = sess.run([merged, tp, train_tp, assign_z1, assign_z2], feed_dict={x:xmb, c:cmb, prob:keep_prob}, options=run_options,run_metadata=run_metadata)
                    if include_cyc:
                        t_loss_list.append(t_loss)
                        p_loss_list.append(p_loss)
                    else:
                        t_loss_list.append(tp_loss)
                        p_loss_list.append(0)
                    #except:
                rec_loss, rec_abs, rec_std = sess.run([recon_loss, recon_abs, recon_std], feed_dict={x:xmb,c:cmb})
                re_loss.append(rec_loss)
                re_abs.append(rec_abs)
                re_std.append(rec_std)
                #train_writer.close()
                y1_ = sess.run(y1, feed_dict={x:xmb,c:cmb}) 
                A_, B_, D_ = sess.run([A, B, D], feed_dict={x:xmb,c:cmb})
                #print ("sTEP:%d si_loss:%f t_loss:%f recon_loss:%f"%(i, s_loss[-1], tp_loss[-1], re_loss[-1]))
                
            if epoch%1000 == 0:
                save_path = saver.save(sess, os.path.join(FLAGS.logdir, "model.ckpt"), epoch)
                print("Model saved at: ", save_path)

            print ("######################## epoch:%d si_loss:%f t_loss:%f p_loss:%f "%(epoch, s_loss[-1], t_loss_list[-1], p_loss_list[-1]))

        y1_out_list = []
        for i in range(100):
            tmp = []
            for j in range(n_train//batch_size):
                y1_ = sess.run(y1, feed_dict={x:XC_dataset[j*batch_size:(j+1)*batch_size,0:inp_data_dim], c:XC_dataset[j*batch_size:(j+1)*batch_size,inp_data_dim:]})
                tmp.append(y1_)
            y1_out_list.append(np.concatenate(tmp, axis=0))
        y1_ = np.array(y1_out_list)
        #y1_ = np.mean(y1_, axis=0)
        y1_out_list = []
        eps2 = standard_normal([test_batch_size, eps2_dim], name="eps2") * 1.0 # (batch_size, eps_dim)
        eps1 = standard_normal([test_batch_size, eps1_dim], name="eps1") * 1.0 # (batch_size, eps_dim)
        noise = standard_normal([test_batch_size, inp_data_dim], name="test_noise") * 1.0 # (batch_size, eps_dim)
        z1_t, z2_t = encoder_network(X_t, C_t, enc_net_hidden_dim, 2, inp_data_dim, latent_dim, eps1, eps2, True, prob)
        y1_t, y2_t, _ , _ , _ = decoder_network(z1_t, z2_t, C_t, True, noise)
        for i in range(100):
            y1t_ = sess.run(y1_t)
            y1_out_list.append(y1t_)
        y1_t_ = np.array(y1_out_list)

        train_writer.close()

       # ################### Adding classifier #########################
       # closs = []
       # train_z2 = []
       # train_z1 = []
       # for i in range(niter_clf):
       #     start = (i*batch_size)%n_train
       #     z1_, z2_ = sess.run([z1,z2], feed_dict={x:XC_dataset[start:start+batch_size,0:inp_data_dim], c:XC_dataset[start:start+batch_size,inp_data_dim:]})
       #     train_z2.append(z2_)
       #     if i == 0:
       #         train_z1.append(z1_)
       #     indices = get_indices()
       #     loss_, _ , out_soft_, out_logits_ = sess.run([loss, train_clf, out_soft, out_logits], feed_dict={z_input:z2_[indices], labels:raw_labels[indices]})
       #     pred_labels = np.argmax(out_soft_, axis=1)
       #     train_accuracy = (pred_labels == raw_labels[indices])
       #     if i%1000 == 0:
       #         print ("train out logits:",out_logits_)
       #         print ("train_indices:", raw_labels[indices])
       #     train_accuracy = np.sum(train_accuracy)/(train_accuracy.shape[0]*1.0)
       #     print("step:%d loss:%f train_accuracy:%f"%(i, loss_, train_accuracy))
       #     closs.append((loss_, train_accuracy))

       # z2t_list = []
       # test_z2 = []
       # for i in range(100):
       #     z2t_ = sess.run(z2_t)
       #     test_z2.append(z2t_)
       #     out_logits_, out_soft_ = sess.run([out_logits, out_soft], feed_dict={z_input:z2t_, labels:test_labels})
       #     assert(out_soft_.shape == (test_batch_size, num_classes))
       #     z2t_list.append(out_soft_)
       # z2t_out = np.array(z2t_list)
       # z2t_out = np.mean(z2t_out, axis=0)        
       # assert(out_soft_.shape == (n_test, num_classes))

       # pred_labels = np.argmax(z2t_out, axis=1)
       # test_accuracy = (pred_labels == test_labels)
       # test_accuracy = np.sum(test_accuracy)
       # print("Test classification Accuracy:", test_accuracy)
       # print ("Test pred labels:",pred_labels)
       # save_path = saver.save(sess, os.path.join(FLAGS.logdir, 'model.ckpt'))
       # print ("Model saved at: ", save_path)
       # ######################################################################


    with open("/opt/data/saket/gene_data/data/y1_elu_100_100_5.pkl", "wb") as pkl_file:
    #    pkl.dump(y1_, pkl_file)
        #np.save("", y1_)
        np.save("/opt/data/saket/gene_data/data/y1_elu_100_100_5.npy", y1_)
        np.save("/opt/data/saket/gene_data/data/y1t.npy", y1_t_)
        #np.save("/opt/data/saket/gene_data/data/train_z2.npy", train_z2)
        #np.save("/opt/data/saket/gene_data/data/test_z2.npy", test_z2)
        #np.save("/opt/data/saket/gene_data/data/train_z1.npy", train_z1)
        np.save("/opt/data/saket/gene_data/data/A_elu_100_100_5.npy", A_)
        np.save("/opt/data/saket/gene_data/data/D_elu_100_100_5.npy", D_)
        np.save("/opt/data/saket/gene_data/data/B_elu_100_100_5.npy", B_)
        #np.save("/opt/data/saket/gene_data/data/closs.npy", closs)
    with open("/opt/data/saket/gene_data/data/s_loss_elu_xavier_100_100_5.pkl", "wb") as pkl_file:
        pkl.dump(s_loss, pkl_file)
    with open("/opt/data/saket/gene_data/data/t_loss_elu_xavier_100_100_5.pkl", "wb") as pkl_file:
        pkl.dump(t_loss_list, pkl_file)
    with open("/opt/data/saket/gene_data/data/p_loss_elu_xavier_100_100_5.pkl", "wb") as pkl_file:
        pkl.dump(p_loss_list, pkl_file)
    with open("/opt/data/saket/gene_data/data/re_loss_elu_xavier_100_100_5.pkl", "wb") as pkl_file:
        pkl.dump(re_loss, pkl_file)
    with open("/opt/data/saket/gene_data/data/acc_elu_xavier_100_100_5.pkl", "wb") as pkl_file:
        pkl.dump(acc, pkl_file)
    with open("/opt/data/saket/gene_data/data/acc_abs_elu_xavier_100_100_5.pkl", "wb") as pkl_file:
        pkl.dump(acc_abs, pkl_file)
    with open("/opt/data/saket/gene_data/data/acc_std_elu_xavier_100_100_5.pkl", "wb") as pkl_file:
        pkl.dump(acc_std, pkl_file)
    with open("/opt/data/saket/gene_data/data/re_abs_loss_elu_xavier_100_100_5.pkl", "wb") as pkl_file:
        pkl.dump(re_abs, pkl_file)
    with open("/opt/data/saket/gene_data/data/re_std_loss_elu_xavier_100_100_5.pkl", "wb") as pkl_file:
        pkl.dump(re_std, pkl_file)

def main():
    tf.reset_default_graph()

    prob = tf.placeholder_with_default(1.0, shape=())
    c = tf.placeholder(tf.float32, shape=(batch_size, inp_cov_dim))
    x = tf.placeholder(tf.float32, shape=(batch_size, inp_data_dim))

    MVN_e2 = ds.MultivariateNormalDiag(tf.zeros((eps2_dim)), tf.ones((eps2_dim)))
    MVN_e1 = ds.MultivariateNormalDiag(tf.zeros((eps1_dim)), tf.ones((eps1_dim)))
    MVN_noise = ds.MultivariateNormalDiag(tf.zeros((inp_data_dim)), tf.ones((inp_data_dim)))
    #eps2 = standard_normal([batch_size, eps2_dim], name="eps2") * 1.0 # (batch_size, eps_dim)
    eps2 = MVN_e2.sample(batch_size)
    #eps1 = standard_normal([batch_size, eps1_dim], name="eps1") * 1.0 # (batch_size, eps_dim)
    eps1 = MVN_e1.sample(batch_size)
    noise = MVN_noise.sample(batch_size)
    z1, z2 = encoder_network(x, c, enc_net_hidden_dim, 2, inp_data_dim, latent_dim, eps1, eps2, False, prob)
    y1, y2, A, B, D = decoder_network(z1, z2, c, False, noise)
    variable_summaries(y1, name="y1_for_input_x")
    z = tf.concat([z1,z2], axis=1)
    z_list = [z]
    y1_list = [y1]
    for i in range(0):
        z1, z2 = encoder_network(x, c, enc_net_hidden_dim, 2, inp_data_dim, latent_dim, eps1, eps2, True, prob)
        # z1, z2 = graph_replace([z1,z2], {x:x,c:c})
        y1, y2, A, B, D = decoder_network(z1, z2, c, True, noise)
        # y1,y2 = graph_replace([y1,y2],{z1:z1,z2:z2})
        z = tf.concat([z1,z2], axis=1)
        z_list.append(z)
        y1_list.append(y1)

    MVN = ds.MultivariateNormalDiag(tf.zeros((latent_dim+inp_data_dim)), tf.ones((latent_dim+inp_data_dim)))
    z_sample_ = MVN.sample(n_samples)
    z_sample_ = tf.Print(z_sample_, [z_sample_], message="Sample z....")
    z_sample = tf.Variable(np.ones((n_samples, latent_dim+inp_data_dim), dtype=np.float32))
    z_assign = tf.assign(z_sample, z_sample_)
    z1_sample = tf.slice(z_sample, [0, 0], [-1, inp_data_dim])
    z2_sample = tf.slice(z_sample, [0, inp_data_dim], [-1, -1])
    z2_sample = transform_z2(z2_sample)
    z_sample = tf.concat([z1_sample, z2_sample], axis=1)
    x_sample, _ , _ , _ , _ = decoder_network(z1_sample, z2_sample, c, True, noise)
    # x_sample = graph_replace(y1, {z1:z1_sample, z2:z2_sample})
    z1_x_e, z2_x_e = encoder_network(x_sample, c, enc_net_hidden_dim, 2, inp_data_dim, latent_dim, eps1, eps2, True, prob)
    # z1_x_e, z2_x_e = graph_replace([z1, z2], {x:x_sample})
    z_x_sample_encoded_list = [tf.concat([z1_x_e, z2_x_e], axis=1)]
    for i in range(0):
        z1_x_e, z2_x_e = encoder_network(x_sample, c, enc_net_hidden_dim, 2, inp_data_dim, latent_dim, eps1, eps2, True, prob)
        # z1_x_e, z2_x_e = graph_replace([z1_x_e, z2_x_e], {x:x_sample})
        z_x_sample_encoded = tf.concat([z1_x_e, z2_x_e], axis=1)
        z_x_sample_encoded_list.append(z_x_sample_encoded)

    ########### Test Dataset

    si, theta, phi = cal_loss(x, z_list, y1_list, x_sample, z_sample, z_x_sample_encoded_list)
    recon_loss = tf.reduce_mean(tf.square(y1-x))
    recon_abs = tf.reduce_mean(tf.abs(y1-x))
    recon_std = tf.sqrt(tf.reduce_mean(tf.square(tf.abs(y1-x) - tf.reduce_mean(tf.abs(y1-x)))))

    z_input = tf.placeholder(dtype = tf.float32, shape = (None, latent_dim))
    labels = tf.placeholder(dtype=tf.int64, shape=(None))
    loss, out_logits, out_soft = classifier(z_input, labels, False)
    train(si, theta, phi, x, c,recon_loss, y1, z1, z2, recon_abs, recon_std, A, B, D, z_assign, z_input, labels, loss, out_logits, out_soft, prob)

main()
