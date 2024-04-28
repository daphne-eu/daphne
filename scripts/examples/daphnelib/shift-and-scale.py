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

# (1) Import DaphneLib.
from daphne.context.daphne_context import DaphneContext
import numpy as np

# (2) Create DaphneContext.
dc = DaphneContext()

# (3) Obtain a DAPHNE matrix.
X = dc.from_numpy(np.random.rand(5, 3))

# (4) Define calculations in DAPHNE.
Y = (X - X.mean(axis=1)) / X.stddev(axis=1) # TODO 0 or 1?

# (5) Compute result in DAPHNE.
print(Y.compute())