import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import pickle as pkl
from tensorflow.contrib.tensorboard.plugins import projector
from sklearn.metrics import roc_auc_score
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
num_classes = 3
inp_data_dim = 10 #d
inp_cov_dim = 10 #d'
latent_dim = 100 #k
batch_size = 160
test_batch_size = 52
eps2_dim = 20
eps1_dim = 10
enc_net_hidden_dim = 512
n_samples = batch_size
n_epoch = 55000
n_clf_epoch = 5000
keep_prob = 1
n_train = 160
n_test = 52
# filename = "M255.pkl"
""" Dataset """
def load_dataset():
    raw_data=np.load('/opt/data/saket/gene_data/data/mod_total_data.npy')
    cov = np.load('/opt/data/saket/gene_data/data/cov.npy')
    labels = np.load('/opt/data/saket/gene_data/data/data_label.npy')
    cov = np.log10(cov+0.1)
    global inp_data_dim
    inp_data_dim = np.shape(raw_data)[1]
    global inp_cov_dim
    inp_cov_dim = np.shape(cov)[1]
    print ("inp_cov_dim:", inp_cov_dim)
    # raw_data,labels, cov = normalize_each_class(raw_data, labels, cov)
    assert(np.shape(raw_data)[0] == np.shape(cov)[0])
    raw_data_mean = np.mean(raw_data, axis=0)
    raw_data = (raw_data-1.0*raw_data_mean)
   # print("raw min",np.min(raw_data))
    cov_data_mean = np.mean(cov, axis=0)
    cov_data = (cov-1.0*cov_data_mean)
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

def add_linear(inputs, targets, activation_fn=None, scope=None,reuse=False):
    """ Ma the input to the target space and do element-wise addition """
    with tf.variable_scope(scope, reuse=reuse):
        t = targets.get_shape().as_list()[1]
        output = slim.fully_connected(inputs, t, activation_fn=None)
        output += targets
    if activation_fn is not None:
        output = activation_fn(output)
    return output

def lrelu(x, leak=0.2, name="lrelu"):
    """ Leaky relu activation function """
    return tf.maximum(x, leak*x)

def encoder_network(x, c, latent_dim, n_layer, z1_dim, z2_dim, eps1, eps2, reuse, prob):
    with tf.variable_scope("encoder", reuse = reuse):
        h = tf.concat([x, c], 1)
        for i in range(n_layer):
            h = slim.fully_connected(h,latent_dim,activation_fn=tf.nn.elu,weights_regularizer=slim.l2_regularizer(0.1))
            if reuse==False and i == 0:
                h = add_linear(eps1, h, activation_fn=tf.nn.elu,scope="encoder_z1",reuse=False)
            else:
                h = add_linear(eps1, h, activation_fn=tf.nn.elu,scope="encoder_z1",reuse=True)
        
        z1 = slim.fully_connected(h, z1_dim, activation_fn=None, biases_initializer=tf.truncated_normal_initializer, weights_regularizer = slim.l2_regularizer(0.1))
        # z1 = tf.nn.dropout(z1, prob)

        h = tf.concat([x, c], axis=1)
        h = add_linear(z1, h, activation_fn=tf.nn.elu, scope="z1_skip", reuse=False)
        for i in range(n_layer):
            h = slim.fully_connected(h, latent_dim, activation_fn=tf.nn.elu, weights_regularizer=slim.l2_regularizer(0.1))
            if reuse==False and i == 0:
                h = add_linear(eps2, h, activation_fn=tf.nn.elu, scope="encoder_z2",reuse=False)
            else:
                h = add_linear(eps2, h, activation_fn=tf.nn.elu, scope="encoder_z2",reuse=True)
                
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

def adversary(x, z, n_layer=2, n_hidden=1024, reuse=False):
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
        z1 = tf.slice(z,[0,0],[-1,inp_data_dim])
        z2 = tf.slice(z,[0,inp_data_dim],[-1,-1])
        # h = tf.concat([x,z], axis=1)
        with tf.variable_scope("x_embed", reuse=reuse):
            h = slim.repeat(x,1,slim.fully_connected,n_hidden,activation_fn=lrelu,weights_regularizer=slim.l2_regularizer(0.1))
            x_embed = slim.fully_connected(h,256,activation_fn=None)

        with tf.variable_scope("z1_embed", reuse=reuse):
            h = slim.repeat(z1, 1, slim.fully_connected, n_hidden, activation_fn=lrelu,weights_regularizer=slim.l2_regularizer(0.1))
            z1_embed = slim.fully_connected(h,256,activation_fn=None)
        
        with tf.variable_scope("z2_embed", reuse=reuse):
            h = slim.repeat(z2, 1, slim.fully_connected, n_hidden,activation_fn=lrelu, weights_regularizer=slim.l2_regularizer(0.1))
            z2_embed = slim.fully_connected(h,256,activation_fn=None)
        h = add_linear(z1_embed,x_embed,activation_fn=lrelu,scope="data_net_z1",reuse=False)
        h = add_linear(z2_embed,h,activation_fn=lrelu,scope="data_net_z2",reuse=False)
               
        h = slim.repeat(h,n_layer,slim.fully_connected,n_hidden,activation_fn=lrelu,weights_regularizer=slim.l2_regularizer(0.1))
        out = slim.fully_connected(h,1,activation_fn=None,weights_regularizer=slim.l2_regularizer(0.1))
    return out

def transform_z2(z2, reuse=False):
    with tf.variable_scope("transform_z2", reuse = reuse):
        h = slim.repeat(z2, 3, slim.fully_connected, 512, activation_fn=lrelu, weights_regularizer = slim.l2_regularizer(0.1))
        h = slim.fully_connected(h,latent_dim, activation_fn=None, weights_regularizer=slim.l2_regularizer(0.1))
    return h

def cal_loss(x, z, y1, x_sample, z_sample, z_e):
    with tf.variable_scope("Loss"):
        Td = adversary(x,z,reuse=False)
        Ti = adversary(x_sample,z_sample,reuse=True)

        #Adversary Accuracy
        correct_labels_adv = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.greater(tf.sigmoid(Td),thresh_adv), tf.int32),1),tf.float32)) + tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.less_equal(tf.sigmoid(Td),thresh_adv), tf.int32),0),tf.float32))
        label_acc_adv = correct_labels_adv/(2.0*batch_size)
        
        # Dual Loss
        d_loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Td, labels=tf.ones_like(Td)))
        d_loss_i = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Ti, labels=tf.zeros_like(Ti)))
        dual_loss = d_loss_d+d_loss_i

        # Primal Loss
        recon_loss_x_z_x = tf.reduce_mean(tf.norm(x-y1, axis=1))
        recon_loss_z_x_z = tf.reduce_mean(tf.norm(z-z_e, axis=1))
        phi = tf.reduce_mean(Td,axis=0)
        theta = -tf.reduce_mean(Ti,axis=0)
        kls = phi+theta
        primal_enc_loss = phi+cyc*recon_loss_x_z_x
        primal_dec_loss = theta+cyc*recon_loss_z_x_z

    return dual_loss, primal_dec_loss, primal_enc_loss, label_acc_adv

def classifier(x_input,labels, reuse=False):
    with tf.variable_scope("Classifier", reuse=reuse):
        logits = slim.fully_connected(x_input, 3, activation_fn=None, weights_regularizer=slim.l2_regularizer(0.1))
        cl_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    return logits, tf.reduce_mean(cl_loss)


def train(primal_dec_loss, primal_enc_loss, dual_loss, label_acc_adv, closs):

    t_vars = tf.trainable_variables()
    evars = [var for var in t_vars if var.name.startswith("encoder")]
    dvars = [var for var in t_vars if var.name.startswith("decoder")]
    lvars = [var for var in t_vars if var.name.startswith("Loss")]
    cvars = [var for var in t_vars if var.name.startswith("Classifier")]
    tz2vars = [var for var in t_vars if var.name.startswith("transform_z2")]

    # Regularisation Loss
    r_loss = tf.losses.get_regularization_loss() 
    r_loss_clf = tf.losses.get_regularization_loss(scope="Classifier")
    r_loss_adv = tf.losses.get_regularization_loss(scope="Loss")
    r_loss = r_loss - r_loss_clf - r_loss_adv

    # Optimiser definition
    primal_dec_opt = tf.train.AdamOptimizer(learning_rate=1e-4, use_locking=True)
    primal_enc_opt = tf.train.AdamOptimizer(learning_rate=1e-4, use_locking=True)
    dual_opt = tf.train.AdamOptimizer(learning_rate=1e-4, use_locking=True)
    classifier_opt = tf.train.AdamOptimizer(learning_rate=1e-4)
    primal_dec_op = primal_dec_opt.minimize(primal_dec_loss+r_loss, var_list=dvars+tz2vars)
    primal_enc_op = primal_enc_opt.minimize(primal_enc_loss+r_loss, var_list=evars)
    dual_op = dual_opt.minimize(dual_loss+r_loss_adv, var_list=lvars)
    train_op = tf.group(primal_dec_op,primal_enc_op,dual_op)
    clf_train_op = classifier_opt.minimize(closs+r_loss_clf, var_list=cvars)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.logdir, graph = tf.get_default_graph())
    config = projector.ProjectorConfig()  
   
    embedding_var = tf.Variable(initial_value = np.zeros((z2.shape), dtype=np.float32), name="z2_embed_var") 
    assign_z2 = tf.assign(embedding_var, z2)
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    projector.visualize_embeddings(train_writer, config)        
    
    dloss_list = []
    ploss_list = []
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer(), feed_dict={x:XC_dataset[0:batch_size,0:inp_data_dim], c:XC_dataset[0:batch_size,inp_data_dim:]})
        saver = tf.train.Saver(save_relative_paths=True)
        # sess.run(z_assign, feed_dict={x:XC_dataset[:,0:inp_data_dim], c:XC_dataset[:,inp_data_dim:]})
        #saver.restore(sess, os.path.join("/opt/data/saket/model_2gg00_100_5/", "model.ckpt-4000"))
        for epoch in range(n_epoch):
            X_dataset = XC_dataset[:,0:inp_data_dim]
            C_dataset = XC_dataset[:,inp_data_dim:]
            sess.run(z_assign, feed_dict={x:XC_dataset[0:batch_size,0:inp_data_dim], c:XC_dataset[0:batch_size,inp_data_dim:], prob:keep_prob})

            for i in range(np.shape(X_dataset)[0]//batch_size):
                xmb = X_dataset[i*batch_size:(i+1)*batch_size]
                cmb = C_dataset[i*batch_size:(i+1)*batch_size]
                _ = sess.run(train_op, feed_dict={x:xmb, c:cmb, prob:keep_prob})
                dual_loss_, primal_dec_loss_, primal_enc_loss_,label_acc_adv_ = sess.run([dual_loss_, primal_dec_loss_, primal_enc_loss_,label_acc_adv],feed_dict={x:xmb, c:cmb, prob:keep_prob})
                if epoch%400 == 0 and i == 0:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary = sess.run(merged,feed_dict={x:xmb,c:cmb})
                    train_writer.add_run_metadata(run_metadata, 'step %d'%(epoch))
                    train_writer.add_summary(summary, epoch)

            dloss_list.append(dual_loss_)
            ploss_list.append((primal_dec_loss_,primal_enc_loss_))
            if epoch%1000 == 0:
                save_path = saver.save(sess, os.path.join(FLAGS.logdir, "model.ckpt"), epoch)
                print("Model saved at: ", save_path)
            if epoch%100 == 0:
                print("epoch:%d adv_loss:%f primal_enc_loss:%f primal_dec_loss:%f adv_accu:%f"%(epoch, dual_loss_,primal_enc_loss_,primal_dec_loss_))

        eps2 = standard_normal([test_batch_size, eps2_dim], name="eps2") * 1.0 # (batch_size, eps_dim)
        eps1 = standard_normal([test_batch_size, eps1_dim], name="eps1") * 1.0 # (batch_size, eps_dim)
        noise = standard_normal([test_batch_size, inp_data_dim], name="test_noise") * 1.0 # (batch_size, eps_dim)
        z1_t, z2_t = encoder_network(X_t, C_t, enc_net_hidden_dim, 2, inp_data_dim, latent_dim, eps1, eps2, True, prob)
        y1_t, y2_t, _ , _ , _ = decoder_network(z1_t, z2_t, C_t, True, noise)

        train_writer.close()

        save_path = saver.save(sess, os.path.join(FLAGS.logdir, 'modelclf.ckpt'))
        print ("Model Saved at:", save_path)
        
        # Traaining Classifier
        clf_loss_list = []
        for i in range(n_clf_epoch):
            for j in range(ntrain//batch_size):
                xmb = X_train[j*batch_size:(j+1)*batch_size]
                cmb = C_train[j*batch_size:(j+1)*batch_size]
                sess.run(clf_train_op, feed_dict={x:xmb,c:cmb,labels:L_train[j*batch_size:(j+1)*batch_size]})           
                cl_loss_, label_acc_ = sess.run([closs, label_acc], feed_dict={x:xmb,c:cmb,labels:L_train[j*batch_size:(j+1)*batch_size]})
            if i%100 == 0:
                print("epoch:",i," closs:",cl_loss_," train_acc:",label_acc_)
            clf_loss_list.append((cl_loss_, label_acc_))

        # Test Set
        logits_test, closs_test = classifier(z2_t,labels,reuse=True)
        prob_test = tf.nn.softmax(logits_test)
        correct_label_pred_test = tf.equal(tf.argmax(logits_test,1),labels)
        label_acc_test = tf.reduce_mean(tf.cast(correct_label_pred_test, tf.float32))        
        label_acc_ = sess.run(label_acc_test, feed_dict={labels:L_test})
        print("Test Set label Accuracy:", label_acc_)
        test_prob = []
        test_acc = []
        for  i in range(250):
            lt, la = sess.run([prob_test, label_acc_test], feed_dict={labels:L_test})
            test_prob.append(lt)
            test_acc.append(la)
        avg_test_prob = np.mean(test_prob,axis=0)
        avg_test_acc1 = np.mean((np.argmax(avg_test_prob,axis=1)==L_test))
        avg_test_acc = np.mean(test_acc)
        print("Average Test Set Accuracy:",avg_test_acc, " :",avg_test_acc1)
        A_, B_, D_ = sess.run([A,B,D])

    np.save("dloss.npy", dloss_list)
    np.save("ploss.npy", ploss_list)
    np.save("clf_loss.npy", clf_loss_list)
    np.save("A.npy",A_)
    np.save("B.npy",B_)
    np.save("delta.npy",D_)
    np.save("test_acc.npy",test_acc)
    np.save("test_prob.npy",test_prob)

if __name__ == "__main__":
    tf.reset_default_graph()

    prob = tf.placeholder_with_default(1.0, shape=())
    c = tf.placeholder(tf.float32, shape=(batch_size, inp_cov_dim))
    x = tf.placeholder(tf.float32, shape=(batch_size, inp_data_dim))
    labels = tf.placeholder(tf.int64, shape=(None))

    # Sample noise and eps for encoder
    MVN_e2 = ds.MultivariateNormalDiag(tf.zeros((eps2_dim)), tf.ones((eps2_dim)))
    MVN_e1 = ds.MultivariateNormalDiag(tf.zeros((eps1_dim)), tf.ones((eps1_dim)))
    MVN_noise = ds.MultivariateNormalDiag(tf.zeros((inp_data_dim)), tf.ones((inp_data_dim)))
    eps2 = MVN_e2.sample(batch_size)
    eps1 = MVN_e1.sample(batch_size)
    noise = MVN_noise.sample(batch_size)

    z1, z2 = encoder_network(x, c, enc_net_hidden_dim, 2, inp_data_dim, latent_dim, eps1, eps2, False, prob)
    y1, y2, A, B, D = decoder_network(z1, z2, c, False, noise)
    variable_summaries(y1, name="y1_for_input_x")
    z = tf.concat([z1,z2], axis=1)
    
    # sample from prior
    MVN = ds.MultivariateNormalDiag(tf.zeros((latent_dim+inp_data_dim)), tf.ones((latent_dim+inp_data_dim)))
    z_sample_ = MVN.sample(n_samples)
    # z_sample_ = tf.Print(z_sample_, [z_sample_], message="Sample z....")
    z_sample = tf.Variable(np.ones((n_samples, latent_dim+inp_data_dim), dtype=np.float32))
    z_assign = tf.assign(z_sample, z_sample_)
    z1_sample = tf.slice(z_sample, [0, 0], [-1, inp_data_dim])
    z2_sample = tf.slice(z_sample, [0, inp_data_dim], [-1, -1])
    z2_sample = transform_z2(z2_sample)
    z_sample = tf.concat([z1_sample, z2_sample], axis=1)
    x_sample, _ , _ , _ , _ = decoder_network(z1_sample, z2_sample, c, True, noise)
    z1_x_e, z2_x_e = encoder_network(x_sample, c, enc_net_hidden_dim, 2, inp_data_dim, latent_dim, np.zeros(eps1.get_shape().as_list()), np.zeros(eps2.get_shape().as_list()), True, prob)
    z_e = tf.concat([z1_x_e,z2_x_e],axis=1)

    dual_loss, primal_dec_loss, primal_enc_loss, label_acc_adv = cal_loss(x,z,y1,x_sample,z_sample,z_e)

    # Classifier
    logits, closs = classifier(z2,labels,reuse=False)
    correct_label_pred = tf.equal(tf.argmax(logits,1),labels)
    label_acc = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))

    train(primal_dec_loss, primal_enc_loss, dual_loss, label_acc_adv, closs)