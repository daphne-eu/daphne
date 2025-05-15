# Data transfer from pandas to DAPHNE and back, via files.  
import pandas as pd
import numpy as np
from daphne.context.daphne_context import DaphneContext

ser1 = pd.Series(np.zeros(1000))

dctx = DaphneContext()

dctx.from_pandas(ser1, shared_memory=False).print().compute(type="files")