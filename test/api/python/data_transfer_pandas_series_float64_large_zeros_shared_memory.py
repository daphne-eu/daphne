# Data transfer from pandas to DAPHNE and back, via shared memory.

import pandas as pd
import numpy as np
from daphne.context.daphne_context import DaphneContext

s = pd.Series(np.zeros(100000))

dctx = DaphneContext()

dctx.from_pandas(s, shared_memory=True).print().compute(type="shared memory")