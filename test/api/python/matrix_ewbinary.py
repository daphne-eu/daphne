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

m1 = dc.fill(2.2, 1, 1)
m2 = dc.fill(3.3, 1, 1)
s2 = 3.3

# matrix op matrix.
(m1 + m2).print().compute()
(m1 - m2).print().compute()
(m1 * m2).print().compute()
(m1 / m2).print().compute()
(m1 ** m2).print().compute()
(m1 % m2).print().compute()
(m1 == m2).print().compute()
(m1 != m2).print().compute()
(m1 < m2).print().compute()
(m1 <= m2).print().compute()
(m1 > m2).print().compute()
(m1 >= m2).print().compute()
(m1 @ m2).print().compute()
m1.pow(m2).print().compute()
m1.log(m2).print().compute()
m1.mod(m2).print().compute()
m1.min(m2).print().compute()
m1.max(m2).print().compute()

# matrix op scalar.
(m1 + s2).print().compute()
(m1 - s2).print().compute()
(m1 * s2).print().compute()
(m1 / s2).print().compute()
(m1 ** s2).print().compute()
(m1 % s2).print().compute()
(m1 == s2).print().compute()
(m1 != s2).print().compute()
(m1 < s2).print().compute()
(m1 <= s2).print().compute()
(m1 > s2).print().compute()
(m1 >= s2).print().compute()
m1.pow(s2).print().compute()
m1.log(s2).print().compute()
m1.mod(s2).print().compute()
m1.min(s2).print().compute()
m1.max(s2).print().compute()

# scalar op matrix.
# TODO