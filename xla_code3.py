import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import Session
from time import process_time

assert (tf.test.is_built_with_cuda())
tf.keras.backend.clear_session()
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
tf.compat.v1.keras.backend.set_session(Session(config=config))
# tf.config.optimizer.set_jit(False)

def model_fn(x, y, z):
  return tf.reduce_sum(x + y * z)

# def create_and_run_graph():
with tf.compat.v1.Session() as sess:
	x = tf.compat.v1.placeholder(tf.float32, name='x')
	y = tf.compat.v1.placeholder(tf.float32, name='y')
	z = tf.compat.v1.placeholder(tf.float32, name='z')
	result = tf.xla.experimental.compile(computation=model_fn, inputs=(x, y, z))[0]
	# `result` is a normal Tensor (albeit one that is computed by an XLA
	# compiled executable) and can be used like any other Tensor.
	result = tf.add(result, result)
	sess.run(result, feed_dict={ ... })
# create_and_run_graph()