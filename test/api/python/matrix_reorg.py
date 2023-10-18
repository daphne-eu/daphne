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

m = dc.seq(1, 6, 1).reshape(3, 2)
s = dc.seq(1, 16, 1).reshape(4, 4)

m.t().print().compute()
m.reshape(3, 2).print().compute()
m.cbind(m).print().compute()
m.rbind(m).print().compute()
m.reverse().print().compute()
s.lowerTri(True, True).print().compute()
s.upperTri(True, True).print().compute()
m.replace(5, -5).print().compute()
# TODO Using false for the last parameter (returnIdxs) crashes (see #612).
# m.order([0, 1], [True, False], False).print().compute()
m.order([0, 1], [True, False], True).print().compute()