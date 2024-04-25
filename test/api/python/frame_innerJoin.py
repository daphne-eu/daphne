#!/usr/bin/python

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

dctx = DaphneContext()

df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
df2 = pd.DataFrame({"c": [3, 4, 5], "d": [6, 7, 8]})

f1 = dctx.from_pandas(df1)
f2 = dctx.from_pandas(df2)

f1.innerJoin(f2, "b", "c").print().compute()