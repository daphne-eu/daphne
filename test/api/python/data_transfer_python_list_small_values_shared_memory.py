# Data transfer from python to DAPHNE and back, via shared memory.

from daphne.context.daphne_context import DaphneContext

m1 = [1e-10, 2e-10, 3e-10]
    
dctx = DaphneContext()

(dctx.from_python(m1, shared_memory=True).print().compute())
