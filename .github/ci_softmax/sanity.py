import numpy as np
import tensorflow as tf
x = tf.random.stateless_normal([2, 3, 4], seed=[1, 2])
U = tf.TensorSpec(None)
ok = True
for f in (tf.nn.softmax, tf.nn.log_softmax):
  for ax in (0, 1, 2, -1, -2, -3):
    ref = f(x, axis=ax).numpy()
    for name, fn, args in [
        ("unknown-rank", tf.function(lambda t, a=ax: f(t, axis=a), input_signature=[U]), (x,)),
        ("tensor-axis", tf.function(lambda t, a: f(t, axis=a)), (x, tf.constant(ax))),
        ("unknown-rank+tensor-axis", tf.function(lambda t, a: f(t, axis=a), input_signature=[U, tf.TensorSpec([], tf.int32)]), (x, tf.constant(ax)))]:
      try:
        good = np.allclose(fn(*args).numpy(), ref)
      except Exception as e:
        good = False
        print(f.__name__, ax, name, type(e).__name__)
      ok &= good
with tf.GradientTape(persistent=True) as tape:
  tape.watch(x)
  w = tf.range(2.)[:, None, None]
  l1 = tf.reduce_sum(tf.function(lambda t: tf.nn.softmax(t, axis=0), input_signature=[U])(x) * w)
  l2 = tf.reduce_sum(tf.nn.softmax(x, axis=0) * w)
try:
  ok &= np.allclose(tape.gradient(l1, x).numpy(), tape.gradient(l2, x).numpy())
except Exception as e:
  ok = False
  print("gradient", type(e).__name__)
try:
  tf.nn.softmax(x, axis=3)
  ok = False
except tf.errors.InvalidArgumentError as e:
  print("known-rank out of range:", e.message)
print("SANITY", "PASS" if ok else "FAIL")
