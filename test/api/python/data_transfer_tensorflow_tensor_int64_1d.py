# Data transfer from tensorflow to DAPHNE and back, via files.

import tensorflow as tf
from daphne.context.daphne_context import DaphneContext

t = tf.constant([0, 1, -1, 12, -12, 1000, -1000], dtype=tf.int64)

dctx = DaphneContext()

dctx.from_tensorflow(t, shared_memory=False).print().compute()