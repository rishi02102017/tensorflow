import traceback
import numpy as np
import tensorflow as tf
print("TF", tf.__version__)
def T(name, fn):
    try:
        r = fn()
        if hasattr(r, 'numpy'): r = r.numpy()
        print(f"[OK ] {name}: {repr(r)[:200]}".replace("\n", " "))
    except Exception as e:
        print(f"[ERR] {name}: {type(e).__name__}: {str(e)[-300:]}".replace("\n", " | "))
D = tf.data.Dataset
T("filter -> [True]", lambda: D.range(3).filter(lambda x: [True]))
T("filter -> (x > 0,)", lambda: D.range(3).filter(lambda x: (x > 0,)))
T("filter -> {'k': x > 0}", lambda: D.range(3).filter(lambda x: {'k': x > 0}))
T("filter -> (x>0, x>1)", lambda: D.range(3).filter(lambda x: (x > 0, x > 1)))
T("filter -> ragged", lambda: D.range(3).filter(lambda x: tf.ragged.constant([[True]])))
T("filter -> None", lambda: D.range(3).filter(lambda x: None))
T("take_while -> (x > 0,)", lambda: D.range(3).take_while(lambda x: (x > 0,)))
T("group_by_window key -> (x,)", lambda: D.range(3).group_by_window(lambda x: (x,), lambda k, w: w, window_size=2))
T("group_by_window wsize -> (x,)", lambda: D.range(3).group_by_window(lambda x: x, lambda k, w: w, window_size_func=lambda k: (k,)))
T("group_by_window key -> int32 (expected ValueError)", lambda: D.range(3).group_by_window(lambda x: tf.cast(x, tf.int32), lambda k, w: w, window_size=2))
T("snapshot shard_func -> (x,)", lambda: D.range(3).snapshot('/tmp/snap', shard_func=lambda x: (x,)))
T("rejection_resample class_func -> (x,)", lambda: D.range(3).rejection_resample(lambda x: (tf.cast(x, tf.int32),), [0.5, 0.5, 0.], seed=1))
# rot90 dead-branch investigation
U = tf.TensorSpec(None)
img = tf.zeros([2, 3, 1])
T("rot90 3D unknown rank k=1", lambda: tf.function(lambda t: tf.image.rot90(t), input_signature=[U])(img).shape)
tf.config.optimizer.set_experimental_options({'disable_meta_optimizer': True})
T("rot90 3D unknown rank k=1 (grappler off)", lambda: tf.function(lambda t: tf.image.rot90(t), input_signature=[U])(img).shape)
tf.config.optimizer.set_experimental_options({'disable_meta_optimizer': False})
T("rot90 3D unknown rank k=tensor", lambda: tf.function(lambda t, k: tf.image.rot90(t, k), input_signature=[U, tf.TensorSpec([], tf.int32)])(img, tf.constant(1)).shape)
print("DONE")
