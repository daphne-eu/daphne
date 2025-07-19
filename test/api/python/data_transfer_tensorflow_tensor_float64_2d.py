# Data transfer from tensorflow to DAPHNE and back, via files.

import numpy as np
import tensorflow as tf
from daphne.context.daphne_context import DaphneContext

t1 = tf.constant([[np.nan, 0.0], [1.0, -1.0], [12.3, -12.3], [2e-10, -2e-10], [2e10, -2e10], [np.inf, -np.inf]], dtype=tf.float64)

dctx = DaphneContext()

(dctx.from_tensorflow(t1, shared_memory=False).print().compute())