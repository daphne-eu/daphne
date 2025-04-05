# Data transfer from pandas to DAPHNE and back, shared memory. 
import numpy as np 
import pandas as pd
from daphne.context.daphne_context import DaphneContext

ser1 = pd.Series([np.nan, np.nan, np.nan])

dctx = DaphneContext()

dctx.from_pandas(ser1, shared_memory=True).print().compute(type="shared memory")