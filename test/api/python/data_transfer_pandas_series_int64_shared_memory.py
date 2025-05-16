# Data transfer from pandas to DAPHNE and back, via shared memory. 

import numpy as np
import pandas as pd
from daphne.context.daphne_context import DaphneContext

s = pd.Series([0, 1, -1, 12, -12, 1000, -1000])

dctx = DaphneContext()

dctx.from_pandas(s, shared_memory=True).print().compute(type="shared memory")