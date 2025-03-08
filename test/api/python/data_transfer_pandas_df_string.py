# Data transfer from pandas to DAPHNE and back, via files.
# pd.DataFrame

import pandas as pd
from daphne.context.daphne_context import DaphneContext

df = pd.DataFrame({"col1": ["red", "green", "blue"], "col2": ["circle", "square", "triangle"]})

dctx = DaphneContext()

dctx.from_pandas(df, shared_memory=False).print().compute(type="files")