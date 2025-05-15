# Data transfer from numpy to DAPHNE and back, via files.

import numpy as np
from daphne.context.daphne_context import DaphneContext

m1 = [1.0, 2.0, 3.0]
m2 = [4.0, 5.0, 6.0, 7.0]
m3 = [8.0, 9.0, 10.0, 11.0, 12.0]
    
dctx = DaphneContext()

(dctx.from_python(m1, shared_memory=True).print().compute())
(dctx.from_python(m2, shared_memory=True).print().compute())
(dctx.from_python(m3, shared_memory=True).print().compute())