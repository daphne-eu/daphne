# Data transfer from python lists to DAPHNE and back, via files.

from daphne.context.daphne_context import DaphneContext

m1 = [["apple", "banana"], ["cherry", "fig"]]
    
dctx = DaphneContext()

(dctx.from_python(m1, shared_memory=False).print().compute())