from api.python.context.daphne_context import DaphneContext
import pandas as pd

dctx = DaphneContext()

df1 = pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})

df2 = pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})

f1 = dctx.from_pandas(df1)
f2 = dctx.from_pandas(df2)

output = dctx.for_loop([f2], lambda x, i: x.rbind(f1), 0, 4)

(output[0]).print().compute()