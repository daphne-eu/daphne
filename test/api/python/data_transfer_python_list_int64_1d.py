# Data transfer from python lists to DAPHNE and back, via files.

from daphne.context.daphne_context import DaphneContext

m = [0, 1, -1, 12, -12, 1000, -1000]

dctx = DaphneContext()

dctx.from_python(m, shared_memory=False).print().compute()