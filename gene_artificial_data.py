import tensorflow as tf
import numpy as np

dist = tf.contrib.distributions
shape = (200,5000)
a = np.random.randint(1,high=1000,size = shape)
b = np.random.randint(1,high=1000, size=shape)

p1 = np.random.uniform(size=shape)
p2 = np.random.uniform(size=shape)

nb1 = dist.NegativeBinomial(a, probs=p1)
nb2 = dist.NegativeBinomial(b, probs=p2)

s1 = nb1.sample()
print("s1 shape:",s1.shape)
