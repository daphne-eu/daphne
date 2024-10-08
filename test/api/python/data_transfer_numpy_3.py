#!/usr/bin/python

# Copyright 2022 The DAPHNE Consortium
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

# Data transfer from numpy to DAPHNE and back, via shared memory.

import numpy as np
from daphne.context.daphne_context import DaphneContext

m1 = np.arange(8).reshape((2,2,2))
m2 = np.arange(27).reshape((3,3,3))
m3 = np.arange(32).reshape((2,2,2,2,2))

dctx = DaphneContext()

X, m1_og_shape = dctx.from_numpy(m1, shared_memory=True, return_shape=True)
Y, m2_og_shape = dctx.from_numpy(m2, shared_memory=True, return_shape=True)
Z, m3_og_shape = dctx.from_numpy(m3, shared_memory=True, return_shape=True)

X.print().compute()
Y.print().compute()
Z.print().compute()
print(m1_og_shape)
print(m2_og_shape)
print(m3_og_shape)
