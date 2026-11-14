import traceback
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import nn_ops, array_ops, math_ops
from tensorflow.python.framework import ops, tensor as tensor_lib, tensor_util, errors_impl
print("TF", tf.__version__)

def T(name, fn, tb=False):
    try:
        r = fn()
        if hasattr(r, 'numpy'): r = r.numpy()
        print(f"[OK ] {name}: {repr(r)[:300]}".replace("\n", " "))
    except Exception as e:
        print(f"[ERR] {name}: {type(e).__name__}: {str(e)[-500:]}".replace("\n", " | "))
        if tb: traceback.print_exc(limit=-4)

U = tf.TensorSpec(None)
x = tf.random.stateless_normal([2, 3, 4], seed=[1, 2])
print("---- minimal repro")
T("softmax axis=1 unknown rank", lambda: tf.function(lambda t: tf.nn.softmax(t, axis=1), input_signature=[U])(x).shape, tb=True)

def run_all(tag):
    for fname, f in [("softmax", tf.nn.softmax), ("log_softmax", tf.nn.log_softmax)]:
        for ax in [0, 1, 2, -1, -2, -3]:
            ref = f(x, axis=ax).numpy()
            T(f"{tag} {fname} axis={ax} unknown-rank", lambda: np.allclose(tf.function(lambda t: f(t, axis=ax), input_signature=[U])(x).numpy(), ref))
            T(f"{tag} {fname} axis=tf.constant({ax}) unknown-rank", lambda: np.allclose(tf.function(lambda t: f(t, axis=tf.constant(ax)), input_signature=[U])(x).numpy(), ref))
            T(f"{tag} {fname} axis={ax} known-rank fn", lambda: np.allclose(tf.function(lambda t: f(t, axis=ax), input_signature=[tf.TensorSpec([None, None, None])])(x).numpy(), ref))
        T(f"{tag} {fname} axis=3 known (expect error)", lambda: f(x, axis=3))
        T(f"{tag} {fname} axis=3 unknown (expect error)", lambda: tf.function(lambda t: f(t, axis=3), input_signature=[U])(x))
        T(f"{tag} {fname} 1-D axis=0 unknown", lambda: tf.function(lambda t: f(t, axis=0), input_signature=[U])(tf.constant([1., 2.])))
        T(f"{tag} {fname} static shape kept", lambda: tf.function(lambda t: f(t, axis=0), input_signature=[U]).get_concrete_function().structured_outputs.shape)
    xi = tf.constant([[3., 1.], [1., 3.], [2., 2.]])
    ref = tf.nn.isotonic_regression(xi, axis=0)[0].numpy()
    T(f"{tag} isotonic axis=0 unknown-rank", lambda: np.allclose(tf.function(lambda t: tf.nn.isotonic_regression(t, axis=0)[0], input_signature=[U])(xi).numpy(), ref))
    T(f"{tag} isotonic axis=-2 unknown-rank", lambda: np.allclose(tf.function(lambda t: tf.nn.isotonic_regression(t, axis=-2)[0], input_signature=[U])(xi).numpy(), ref))

T("tensor axis known-rank in tf.function (no unknown rank)", lambda: tf.function(lambda t: tf.nn.softmax(t, axis=tf.constant(0)))(x).shape)
T("tensor axis known-rank in tf.function -1", lambda: tf.function(lambda t: tf.nn.softmax(t, axis=tf.constant(-1)))(x).shape)
T("placeholder axis known-rank in tf.function", lambda: tf.function(lambda t, a: tf.nn.softmax(t, axis=a))(x, tf.constant(1)).shape)
T("tensor axis eager", lambda: tf.nn.softmax(x, axis=tf.constant(0)).shape)
run_all("BEFORE")

def _wrap_2d_function_fixed(inputs, compute_op, dim=-1, name=None):
  def _swap_axis(input_tensor, dim_index, last_index, name=None):
    perm = math_ops.range(last_index + 1)
    perm = array_ops.where_v2(math_ops.equal(perm, dim_index), last_index,
                           array_ops.where_v2(math_ops.equal(perm, last_index), dim_index, perm))
    return array_ops.transpose(input_tensor, perm, name=name)

  inputs = ops.convert_to_tensor(inputs)
  shape = inputs.get_shape()
  ndims = shape.ndims
  dim_val = dim
  if isinstance(dim, tensor_lib.Tensor):
    dim_val = tensor_util.constant_value(dim)
  if dim_val is not None and ndims is not None and not -ndims <= dim_val < ndims:
    raise errors_impl.InvalidArgumentError(None, None, f"`dim` must be in the range [{-ndims}, {ndims}) ... Received: dim={dim_val}")
  if dim_val is not None and (dim_val == -1 or (ndims is not None and dim_val == ndims - 1)):
    return compute_op(inputs, name=name)
  input_rank = array_ops.rank(inputs)
  if dim_val is not None and ndims is not None:
    dim_axis = int(dim_val) % ndims
  else:
    dim = math_ops.cast(dim, input_rank.dtype)
    dim_axis = array_ops.where_v2(math_ops.less(dim, 0), dim + input_rank, dim)
  last = math_ops.subtract(input_rank, 1)
  inputs = _swap_axis(inputs, dim_axis, last)
  def fix_output(output):
    output = _swap_axis(output, dim_axis, last, name=name)
    output.set_shape(shape)
    return output
  outputs = compute_op(inputs)
  if isinstance(outputs, tuple):
    return tuple(fix_output(o) for o in outputs)
  return fix_output(outputs)

nn_ops._wrap_2d_function = _wrap_2d_function_fixed
run_all("AFTER")
T("AFTER tensor axis known-rank in tf.function", lambda: tf.function(lambda t: tf.nn.softmax(t, axis=tf.constant(0)))(x).shape)
T("AFTER placeholder axis known-rank in tf.function", lambda: np.allclose(tf.function(lambda t, a: tf.nn.softmax(t, axis=a))(x, tf.constant(1)).numpy(), tf.nn.softmax(x, axis=1).numpy()))
T("AFTER placeholder axis=-1 unknown", lambda: np.allclose(tf.function(lambda t, a: tf.nn.softmax(t, axis=a), input_signature=[U, tf.TensorSpec([], tf.int32)])(x, tf.constant(-1)).numpy(), tf.nn.softmax(x).numpy()))
T("AFTER placeholder axis=2 unknown", lambda: np.allclose(tf.function(lambda t, a: tf.nn.softmax(t, axis=a), input_signature=[U, tf.TensorSpec([], tf.int32)])(x, tf.constant(2)).numpy(), tf.nn.softmax(x).numpy()))
def g():
  with tf.GradientTape() as tape:
    tape.watch(x)
    y = tf.function(lambda t: tf.nn.softmax(t, axis=0), input_signature=[U])(x)
    l = tf.reduce_sum(y * tf.range(2.)[:, None, None])
  g1 = tape.gradient(l, x)
  with tf.GradientTape() as tape:
    tape.watch(x)
    l = tf.reduce_sum(tf.nn.softmax(x, axis=0) * tf.range(2.)[:, None, None])
  return np.allclose(g1.numpy(), tape.gradient(l, x).numpy())
T("AFTER gradient matches", g)
print("DONE")
