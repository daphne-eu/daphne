# Data transfer from pandas to DAPHNE and back, via files. 

import pandas as pd
from daphne.context.daphne_context import DaphneContext

ser1 = pd.Series([10, 12, 14])
ser2 = pd.Series([16, 18, 20, 22])

dctx = DaphneContext()

dctx.from_pandas(ser1, shared_memory=False).print().compute(type="files")
dctx.from_pandas(ser2, shared_memory=False).print().compute(type="files")