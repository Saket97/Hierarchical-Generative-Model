import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC

plt.switch_backend("agg")
inp_dim = 100
num_classes = 2
niter = 1000
reg  = 0.01

def get_indices(raw_labels):
#    i2 = (raw_labels==2).nonzero()[0]
    i1 = (raw_labels==1).nonzero()[0]
    i0 = (raw_labels==0).nonzero()[0]
    a = np.amin([i1.shape[0],i0.shape[0]])
#    np.random.shuffle(i2) 
    np.random.shuffle(i1) 
    np.random.shuffle(i0)
    r = np.concatenate((i1[0:a], i0[0:a]))
    np.random.shuffle(r)
    return r

#def SVM_C():
#    train_z2 = np.load("/opt/data/saket/gene_data/data/train_z2.npy")
#    test_z2 = np.load("/opt/data/saket/gene_data/data/test_z2.npy")
#    labels = np.load("/opt/data/saket/gene_data/data/data_label.npy")
#    i2 = (labels==2).nonzero()[0]
#    i1 = (labels==1).nonzero()[0]
#    i0 = (labels==0).nonzero()[0]
#    labels[i1] = 1
#    labels[i2] = 0
#    labels[i0] = 0
#    train_labels = labels[0:160]
#    test_labels = labels[160:]
#    clf = SVC(probability=True)
#    train_z2 = np.mean(train_z2, axis=0)
#    clf.fit(train_z2, train_labels)
#    test_z2 = np.mean(test_z2, axis=0)
#    metrics = clf.predict(test_z2)
#    Accuracy = 0
#    n = metrics.shape[0]
#    for i in range(n):
#        if metrics[i] == test_labels[i]:
#            Accuracy += 1
#    print ("Accuracy:",Accuracy)

def normalize(x):
    m,v = tf.nn.moments(x, [0])
    y = (x-m)/tf.sqrt(v)
    return y
def main():
    
    x = tf.placeholder(tf.float32, [None, inp_dim])
    y1 = normalize(x)
    W = tf.Variable(tf.ones([inp_dim, num_classes]))
    b = tf.Variable(tf.zeros([num_classes]))
    y = tf.matmul(y1,W)+b
    ys = tf.nn.softmax(y)
    y_ = tf.placeholder(tf.int64, [None])

    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy+reg*tf.nn.l2_loss(W))
    
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    
    train_z2 = np.load("/opt/data/saket/gene_data/data/train_z2.npy")
    test_z2 = np.load("/opt/data/saket/gene_data/data/test_z2.npy")
    labels = np.load("/opt/data/saket/gene_data/data/data_label.npy")
    #train_z2 = train_z2*(10**5)
    #test_z2 = test_z2*(10**5)
    print ("Class 0 Vs all")
    i2 = (labels==2).nonzero()[0]
    i1 = (labels==1).nonzero()[0]
    i0 = (labels==0).nonzero()[0]
    labels[i1] = 0
    labels[i2] = 0
    labels[i0] = 1

    train_labels = labels[0:160]
    test_labels = labels[160:]
        
    np.random.shuffle(train_z2)
    closs = []
    correct_prediction = tf.equal(tf.argmax(y,1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    for i in range(niter):
        z2 = train_z2[i%train_z2.shape[0]]
        #print ("z2:",z2)
        indices = get_indices(train_labels)
#        print("normalized data:",sess.run(y1, feed_dict={x:z2[indices], y_:train_labels[indices]}))
        l, train_a, _ = sess.run([cross_entropy, accuracy, train_step], feed_dict = {x:z2[indices], y_:train_labels[indices]})
        closs.append((l, train_a))
     #   print ("Step:%d accuracy:%f loss:%f"%(i, train_a, l))

    # Test the trained model
    testylist = []
    for i in range(test_z2.shape[0]):
        y = sess.run(ys, feed_dict={x:test_z2[i], y_:test_labels})
        testylist.append(y)
        #print(sess.run(y1, feed_dict={x:test_z2[i], y_:test_labels}))
    testy = np.array(testylist)
    testy = np.mean(testy, axis=0)
    pred_labels = np.argmax(testy, axis=1)
    print ("pred_labels:", pred_labels)
#    indices = np.concatenate((np.expand_dims(np.arange(0,pred_labels.shape[0]), axis=1), np.expand_dims(pred_labels, axis=1)), axis=1)
    score = testy[np.arange(0,pred_labels.shape[0]),pred_labels]
    print("indices:", indices.shape)
    print("score shape:",score.shape)
    print("test_labels_shape:", test_labels.shape)
    print("ROC Score:", roc_auc_score(test_labels, score))    
    n = pred_labels.shape[0]
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(n):
        if test_labels[i] == 0:
            if pred_labels[i] == 0:
                TN += 1
            else:
                FP += 1
        else:
            if pred_labels[i] == 0:
                FN += 1
            else:
                TP += 1
    test_accuracy = pred_labels==test_labels
    test_accuracy = np.sum(test_accuracy)
    print("Test classification accuracy:",test_accuracy)
    closs = np.array(closs)
    plt.plot(closs[:,0])
    plt.savefig("closs.png")
    plt.close()
    print("Test FP:%d TP:%d FN:%d TN:%d"%(FP,TP,FN,TN))

main()
