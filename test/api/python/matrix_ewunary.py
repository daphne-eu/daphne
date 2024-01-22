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

from daphne.context.daphne_context import DaphneContext

dc = DaphneContext()

dc.fill(1, 1, 1).abs().print().compute()
dc.fill(0, 1, 1).abs().print().compute()
dc.fill(-3.3, 1, 1).abs().print().compute()

dc.fill(1, 1, 1).sign().print().compute()
dc.fill(0, 1, 1).sign().print().compute()
dc.fill(-3.3, 1, 1).sign().print().compute()

m = dc.fill(0.99, 1, 1)

m.exp().print().compute()
m.ln().print().compute()
m.sqrt().print().compute()

m.round().print().compute()
m.floor().print().compute()
m.ceil().print().compute()

m.sin().print().compute()
m.cos().print().compute()
m.tan().print().compute()
m.asin().print().compute()
m.acos().print().compute()
m.atan().print().compute()
m.sinh().print().compute()
m.cosh().print().compute()
m.tanh().print().compute()