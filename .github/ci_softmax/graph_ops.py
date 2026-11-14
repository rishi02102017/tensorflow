"""Prints the ops _wrap_2d_function emits on the common static path."""
import tensorflow as tf
for axis in (0, 1, -2, -1):
  fn = tf.function(lambda x, a=axis: tf.nn.softmax(x, axis=a))
  g = fn.get_concrete_function(tf.TensorSpec([2, 3, 4])).graph
  print("axis", axis, [op.type for op in g.get_operations()])
