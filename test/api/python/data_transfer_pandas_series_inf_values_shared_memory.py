# Data transfer from pandas to DAPHNE and back, via shared memory. 
import numpy as np 
import pandas as pd
from daphne.context.daphne_context import DaphneContext

ser1 = pd.Series([np.inf, -np.inf, np.inf])

dctx = DaphneContext()

dctx.from_pandas(ser1, shared_memory=True).print().compute(type="shared memory")