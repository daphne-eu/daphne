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

# TODO Currently, we cannot simply construct a DaphneLib scalar from a Python scalar.
# Thus, we use a work-around here by taking the sum of a 1x1 matrix with the desired value.

dc.fill(1, 1, 1).sum().abs().print().compute()
dc.fill(0, 1, 1).sum().abs().print().compute()
dc.fill(-3.3, 1, 1).sum().abs().print().compute()

dc.fill(1, 1, 1).sum().sign().print().compute()
dc.fill(0, 1, 1).sum().sign().print().compute()
dc.fill(-3.3, 1, 1).sum().sign().print().compute()

s = dc.fill(0.99, 1, 1)

s.sum().exp().print().compute()
s.sum().ln().print().compute()
s.sum().sqrt().print().compute()

s.sum().round().print().compute()
s.sum().floor().print().compute()
s.sum().ceil().print().compute()

s.sum().sin().print().compute()
s.sum().cos().print().compute()
s.sum().tan().print().compute()
s.sum().asin().print().compute()
s.sum().acos().print().compute()
s.sum().atan().print().compute()
s.sum().sinh().print().compute()
s.sum().cosh().print().compute()
s.sum().tanh().print().compute()