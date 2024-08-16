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
import numpy as np

dc = DaphneContext()

# Create data in numpy.
a = np.arange(8.0).reshape((2, 4))

# Transfer data to DaphneLib (lazily evaluated).
X = dc.from_numpy(a)

print("How DAPHNE sees the data from numpy:")
X.print().compute()

# Add 100 to each value in X.
X = X + 100.0

# Compute in DAPHNE, transfer result back to Python.
print("\nResult of adding 100 to each value, back in Python:")
print(X.compute())