# Data transfer from python lists to DAPHNE and back, via shared memory.

from daphne.context.daphne_context import DaphneContext

m = []
for i in range(0, 1000):
    m.append([])
    for j in range(0, 1000):
        m[i].append(1.0)


dctx = DaphneContext()

dctx.from_python(m, shared_memory=True).print().compute()