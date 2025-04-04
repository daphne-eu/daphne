# Data transfer from pandas to DAPHNE and back, via shared memory.  
import pandas as pd
from daphne.context.daphne_context import DaphneContext

ser1 = pd.Series([1e-10, 2e-10, 3e-10])

dctx = DaphneContext()

dctx.from_pandas(ser1, shared_memory=True).print().compute(type="shared_memory")