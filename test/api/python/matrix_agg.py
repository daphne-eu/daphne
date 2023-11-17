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

dc = DaphneContext()

m = dc.seq(1, 12, 1).reshape(4, 3)


# Full aggregation.
m.sum().print().compute()
m.mean().print().compute()
m.var().print().compute()
m.stddev().print().compute()
m.aggMin().print().compute()
m.aggMax().print().compute()

# Row-wise aggregation.
m.sum(axis=0).print().compute()
m.mean(axis=0).print().compute()
m.var(axis=0).print().compute()
m.stddev(axis=0).print().compute()
m.aggMin(axis=0).print().compute()
m.aggMax(axis=0).print().compute()
m.idxMin(axis=0).print().compute()
m.idxMax(axis=0).print().compute()

# Column-wise aggregation.
m.sum(axis=1).print().compute()
m.mean(axis=1).print().compute()
m.var(axis=1).print().compute()
m.stddev(axis=1).print().compute()
m.aggMin(axis=1).print().compute()
m.aggMax(axis=1).print().compute()
m.idxMin(axis=1).print().compute()
m.idxMax(axis=1).print().compute()

# Cumulative (column-wise) aggregation.
m.cumSum().print().compute()
m.cumProd().print().compute()
m.cumMin().print().compute()
m.cumMax().print().compute()