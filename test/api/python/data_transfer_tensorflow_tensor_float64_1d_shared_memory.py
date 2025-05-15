# Data transfer from tensorflow to DAPHNE and back, via files.

import tensorflow as tf
from daphne.context.daphne_context import DaphneContext

t1 = tf.constant([1.0, 2.0, 3.0], dtype=tf.float64)
t2 = tf.constant([4.0, 5.0, 6.0, 7.0], dtype=tf.float64)
t3 = tf.constant([8.0, 9.0, 10.0, 11.0, 12.0], dtype=tf.float64)

dctx = DaphneContext()

(dctx.from_tensorflow(t1, shared_memory=True).print().compute())
(dctx.from_tensorflow(t2, shared_memory=True).print().compute())
(dctx.from_tensorflow(t3, shared_memory=True).print().compute())