import cv2
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import Session 
from time import process_time

assert (tf.test.is_built_with_cuda())
tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(False)
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
tf.compat.v1.keras.backend.set_session(Session(config=config))

image1 = cv2.imread('/home/mmadala/tmp/scripts/image.png',0)
image2 = cv2.imread('/home/mmadala/tmp/scripts/image2.png',0)

image1 = cv2.resize(image1,(512,512))
#print(image1.shape,image2.shape)

noXLA_start=process_time()

with Session() as sess:
    x = tf.compat.v1.placeholder(tf.float32, name='x')
    y = tf.compat.v1.placeholder(tf.float32, name='y')
    z = tf.compat.v1.placeholder(tf.float32, name='z')
    tf_result = tf.reduce_sum(x + y * z)
    result = tf.add(tf_result, tf_result)
    output = sess.run(result, feed_dict={ x:image1,y:image2,z:image1 })
    print(output)
noXLA_end=process_time()
print("Elapsed time without XLA ",noXLA_end - noXLA_start)
tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(True)
XLA_start=process_time()
with Session() as sess:
    x = tf.compat.v1.placeholder(tf.float32, name='x')
    y = tf.compat.v1.placeholder(tf.float32, name='y')
    z = tf.compat.v1.placeholder(tf.float32, name='z')
    tf_result = tf.reduce_sum(x + y * z)
    result = tf.add(tf_result, tf_result)
    output = sess.run(result, feed_dict={ x:image1,y:image2,z:image1 })
    print(output)
XLA_end=process_time()
print("Elapsed time with XLA ",XLA_end - XLA_start)