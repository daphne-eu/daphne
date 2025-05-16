# Data transfer from pandas to DAPHNE and back, via files.

import pandas as pd
import numpy as np
from daphne.context.daphne_context import DaphneContext

s = pd.Series(np.ones(100000))

dctx = DaphneContext()

dctx.from_pandas(s, shared_memory=False).print().compute(type="files")