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

import pandas as pd
from api.python.context.daphne_context import DaphneContext

dc = DaphneContext()

f1 = dc.from_pandas(pd.DataFrame({"a": [3, 1, 3, 2], "b": [3.3, 1.1, 3.0, 2.2]}))
f2 = dc.from_pandas(pd.DataFrame({"c": [-10, -20, -30, -40]}))

f1.cbind(f2).print().compute()
f1.rbind(f1).print().compute()
# TODO Using false for the last parameter (returnIdxs) crashes (see #612).
# f1.order([0, 1], [True, False], False).print().compute()
f1.order([0, 1], [True, False], True).print().compute()