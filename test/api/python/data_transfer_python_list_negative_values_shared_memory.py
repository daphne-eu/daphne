# Data transfer from python lists to DAPHNE and back, via shared memory.

from daphne.context.daphne_context import DaphneContext

m1 = [-1.0, -2.0, -3.0]
    
dctx = DaphneContext()

(dctx.from_python(m1, shared_memory=True).print().compute())
