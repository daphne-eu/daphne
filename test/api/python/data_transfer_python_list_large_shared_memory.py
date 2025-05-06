# Data transfer from python lists to DAPHNE and back, via shared memory.

from daphne.context.daphne_context import DaphneContext

m1 = []

for i in range(0, 1000):
    m1.append([])
    for j in range(0, 1000):
        m1[i].append(j)


dctx = DaphneContext()

dctx.from_python(m1, shared_memory=True).compute()