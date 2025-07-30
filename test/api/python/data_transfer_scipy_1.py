import numpy as np
import scipy.sparse as sp
from daphne.context.daphne_context import DaphneContext


# CSRMatrix(3x3, int64_t)
# 0 0 0
# 0 0 3
# 0 0 0

data = np.array([3], dtype=np.int64)
indices = np.array([2], dtype=np.uintp)
indptr = np.array([0, 1, 1, 1], dtype=np.uintp)
shape = (3, 3)

csr = sp.csr_matrix((data, indices, indptr), shape=shape, dtype=np.int64)

dctx = DaphneContext()

A, s1 = dctx.from_scipy(csr, shared_memory=True, return_shape=True)
A.print().compute()
