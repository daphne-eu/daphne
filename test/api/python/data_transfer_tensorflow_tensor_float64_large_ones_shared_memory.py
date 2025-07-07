# Data transfer from tensorflow to DAPHNE and back, via shared memory.

import numpy as np
import tensorflow as tf
from daphne.context.daphne_context import DaphneContext

t = tf.constant(np.ones((1000, 1000), dtype=np.float64))

dctx = DaphneContext()

dctx.from_tensorflow(t, shared_memory=True).print().compute()