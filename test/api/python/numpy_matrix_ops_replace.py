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

import numpy as np 
from daphne.context.daphne_context import DaphneContext

m = np.array([1, 2, 3, 0, 0, 0, 7, 8, 9], dtype=np.int64)
m.shape = (3, 3)

dctx = DaphneContext()

M = dctx.from_numpy(m)

M = M.replace(0, 10)

M.print().compute()