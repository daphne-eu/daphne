# Data transfer from tensorflow to DAPHNE and back, via files.

import tensorflow as tf
from daphne.context.daphne_context import DaphneContext

t1 = tf.constant([1e-10, 2e-10, 3e-10], dtype=tf.float64)

dctx = DaphneContext()

(dctx.from_tensorflow(t1, shared_memory=True).print().compute())
