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

m1 = dc.fill(1.23, 1, 1)
m2 = dc.fill(4.56, 1, 1)

m1.outerAdd(m2).print().compute()
m1.outerSub(m2).print().compute()
m1.outerMul(m2).print().compute()
m1.outerDiv(m2).print().compute()
m1.outerPow(m2).print().compute()
m1.outerLog(m2).print().compute()
m1.outerMod(m2).print().compute()
m1.outerMin(m2).print().compute()
m1.outerMax(m2).print().compute()
m1.outerAnd(m2).print().compute()
m1.outerOr(m2).print().compute()
# TODO Support this op.
# m1.outerXor(m2).print().compute()
# TODO Support this op.
# m1.outerConcat(m2).print().compute()
m1.outerEq(m2).print().compute()
m1.outerNeq(m2).print().compute()
m1.outerLt(m2).print().compute()
m1.outerLe(m2).print().compute()
m1.outerGt(m2).print().compute()
m1.outerGe(m2).print().compute()