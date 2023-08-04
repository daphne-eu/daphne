# Copyright 2023 The DAPHNE Consortium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from api.python.context.daphne_context import DaphneContext
import pandas as pd

dc = DaphneContext()

# Create data in pandas.
df = pd.DataFrame({"a": [1, 2, 3], "b": [1.1, -2.2, 3.3]})

# Transfer data to DaphneLib
F = dc.from_pandas(df)

#print(F)

print("How DAPHNE sees the data from pandas:")
F.print().compute()

# Append F to itself.
F1 = F.rbind(F)

# Compute in DAPHNE, transfer result back to Python.
print("\nResult of appending the frame to itself, back in Python:")
print(F1.compute())


print("\n\nShowcasing Indices in Daphne")
# Create data in pandas.
df2 = pd.DataFrame(
    {"c": [30, 25, 0], 
     "d": [ -1, 2.4, 5.9], 
     "e": [7, 5, 9]})

# Transfer data to DaphneLib with index
F2 = dc.from_pandas(df2, keepIndex=True)

print("How DAPHNE sees the data from pandas:")
F2.print().compute(useIndexColumn=True)

# Append F to F2 
F2 = F2.cbind(F)

# Compute in DAPHNE, transfer result back to Python.
print("\nResult of appending both frames horizontally, back in Python:")
print(F2.compute(useIndexColumn=True))