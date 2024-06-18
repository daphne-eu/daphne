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

import math
from daphne.context.daphne_context import DaphneContext

dc = DaphneContext()

arg_m_1 = dc.seq(-2, 5).reshape(2, 4)
info_m = dc.seq(-1, 0).rbind(dc.seq(5, 6)).reshape(1, 4)
arg_m_1.oneHot(info_m).print().compute()

arg_m_2 = dc.seq(10, 70, 10).reshape(1, 7)
arg_m_2.bin(3).print().compute()
arg_m_2.bin(3, 10, 70).print().compute()

arg_m_3 = dc.seq(5.0, 20.0, 15).rbind(dc.fill(math.nan, 1, 1)).rbind(dc.fill(40.0, 1, 1)).rbind(dc.fill(math.inf, 1, 1)).rbind(dc.seq(60.0, 100.0, 40))
arg_m_3.reshape(1, 7).bin(3, 10.0, 70.0).print().compute()
