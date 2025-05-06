# Data transfer from tensorflow to DAPHNE and back, via files.

import numpy as np
import tensorflow as tf
from daphne.context.daphne_context import DaphneContext

t1 = tf.constant(np.ones((1000, 1000), dtype=np.float64))

dctx = DaphneContext()

dctx.from_tensorflow(t1, shared_memory=True).compute()