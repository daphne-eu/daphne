from daphne.context.daphne_context import DaphneContext

dctx = DaphneContext()

csr = (dctx.rand(10**6, 10**6, 0, 3, 0.0001, 912)).compute()
print(csr.shape)
print(csr.nnz)