# Data transfer from python lists to DAPHNE and back, via files.

from daphne.context.daphne_context import DaphneContext

m = [["apple", "banana"], ["cherry", "fig"]]
    
dctx = DaphneContext()

(dctx.from_python(m, shared_memory=False).print().compute())