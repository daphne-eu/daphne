# Data transfer from python lists to DAPHNE and back, via files.

from daphne.context.daphne_context import DaphneContext

m1 = [[1.0, 2.0], [3.0, 4.0]]
m2 = [[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]
m3 = [[11.0, 12.0], [13.0, 14.0], [15.0, 16.0]]
m4 = [[17.0, 18.0, 19.0], [20.0, 21.0, 22.0], [23.0, 24.0, 25.0]]
    
dctx = DaphneContext()

(dctx.from_python(m1, shared_memory=False).print().compute())
(dctx.from_python(m2, shared_memory=False).print().compute())
(dctx.from_python(m3, shared_memory=False).print().compute())
(dctx.from_python(m4, shared_memory=False).print().compute())