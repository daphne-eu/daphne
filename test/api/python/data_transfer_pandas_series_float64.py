# Data transfer from pandas to DAPHNE and back, via files. 

import numpy as np
import pandas as pd
from daphne.context.daphne_context import DaphneContext

s = pd.Series([np.nan, 0.0, 1.0, -1.0, 12.3, -12.3, 2e-10, -2e-10, 2e10, -2e10, np.inf, -np.inf])

dctx = DaphneContext()

dctx.from_pandas(s, shared_memory=False).print().compute(type="files")