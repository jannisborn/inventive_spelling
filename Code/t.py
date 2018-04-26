import warnings
warnings.filterwarnings("ignore",category=FutureWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from time import time
from tensorflow.python.client import device_lib



t = time()
a = tf.random_uniform((1000,10000))
b = tf.random_uniform((10000,1000))

c = tf.matmul(a,b)

with tf.Session() as sess:

	res = sess.run(c)
print(res.shape)
print(time()-t)



print(device_lib.list_local_devices())

