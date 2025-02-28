import tensorflow as tf
from daphne.context.daphne_context import DaphneContext

X1 = tf.constant([1, 2, 3], dtype=tf.float64)
X2 = tf.constant([4, 5, 6], dtype=tf.float64)

dctx = DaphneContext()

X1_daphne = dctx.from_tensorflow(X1, shared_memory=False)
X2_daphne = dctx.from_tensorflow(X2, shared_memory=False)

X1c = X1_daphne.reshape(X1_daphne.ncell(), 1)
X2c = X2_daphne.reshape(X2_daphne.ncell(), 1)

Y = X1c.rbind(X2c)

Y.print().compute()

