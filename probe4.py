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

run_all("BEFORE")
print("DONE")
