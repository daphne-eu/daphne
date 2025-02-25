# Data transfer from tensorflow to DAPHNE and back, via files.

import tensorflow as tf
import numpy as np
from daphne.context.daphne_context import DaphneContext

t1 = tf.constant([np.inf, -np.inf, np.inf], dtype=tf.float64)
    
dctx = DaphneContext()

(dctx.from_tensorflow(t1, shared_memory=False).print().compute())
