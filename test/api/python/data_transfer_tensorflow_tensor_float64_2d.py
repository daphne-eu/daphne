# Data transfer from tensorflow to DAPHNE and back, via files.

import tensorflow as tf
from daphne.context.daphne_context import DaphneContext

t1 = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float64)
t2 = tf.constant([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]], dtype=tf.float64)
t3 = tf.constant([[11.0, 12.0], [13.0, 14.0], [15.0, 16.0]], dtype=tf.float64)
t4 = tf.constant([[17.0, 18.0, 19.0], [20.0, 21.0, 22.0], [23.0, 24.0, 25.0]], dtype=tf.float64)

dctx = DaphneContext()

(dctx.from_tensorflow(t1, shared_memory=False).print().compute())
(dctx.from_tensorflow(t2, shared_memory=False).print().compute())
(dctx.from_tensorflow(t3, shared_memory=False).print().compute())
(dctx.from_tensorflow(t4, shared_memory=False).print().compute())