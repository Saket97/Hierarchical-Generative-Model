import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("agg")
def sample_gumbel(shape, eps=1e-20):
    """ Sample from Gumbel(0,1)"""
    U = tf.random_uniform(shape, minval=0,maxval=1)
    return -tf.log(-tf.log(U+eps) + eps)

def gumble_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution """
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y/temperature)

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
    y = tf.Print(y,[y])
    return y

t = [0.1,0.2,0.5,0.8,1.5,1.8,3,4.5,6,10]
l = np.ones((1,2),dtype=np.float32)
logits = tf.placeholder(dtype=tf.float32,shape=(1,2))
temp = tf.placeholder(dtype=tf.float32,shape=())
sess = tf.Session()
sess.run(tf.global_variables_initializer())
out_list = []
out = gumbel_softmax(logits,temp)
out = tf.transpose(out)
for i in range(len(t)):
    out_ = sess.run(out,feed_dict={logits:l,temp:t[i]})
    print(out_)
    plt.figure()
    plt.hist(out_, bins=30)
    plt.savefig("g%f.png"%t[i])
    plt.close()
    out_list.append(out_)

