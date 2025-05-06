# Data transfer from tensorflow to DAPHNE and back, via files.

import numpy as np
import tensorflow as tf
from daphne.context.daphne_context import DaphneContext

t1 = tf.constant([np.nan, np.nan, np.nan], dtype=tf.float64)

dctx = DaphneContext()

(dctx.from_tensorflow(t1, shared_memory=False).print().compute())
