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
import torch
import numpy as np

dc = DaphneContext()

print("========== 2D TENSOR EXAMPLE ==========\n")

# Create data in PyTorch/numpy.
t2d = torch.tensor(np.random.random(size=(2, 4)))

print("Original 2d tensor in PyTorch:")
print(t2d)

# Transfer data to DaphneLib (lazily evaluated).
T2D = dc.from_pytorch(t2d)

print("\nHow DAPHNE sees the 2d tensor from PyTorch:")
T2D.print().compute()

# Add 100 to each value in T2D.
T2D = T2D + 100.0

# Compute in DAPHNE, transfer result back to Python.
print("\nResult of adding 100, back in Python:")
print(T2D.compute(asPyTorch=True))

print("\n========== 3D TENSOR EXAMPLE ==========\n")

# Create data in PyTorch/numpy.
t3d = torch.tensor(np.random.random(size=(2, 2, 2)))

print("Original 3d tensor in PyTorch:")
print(t3d)

# Transfer data to DaphneLib (lazily evaluated).
T3D, T3D_shape = dc.from_pytorch(t3d, return_shape=True)

print("\nHow DAPHNE sees the 3d tensor from PyTorch:")
T3D.print().compute()

# Add 100 to each value in T3D.
T3D = T3D + 100.0

# Compute in DAPHNE, transfer result back to Python.
print("\nResult of adding 100, back in Python:")
print(T3D.compute(asPyTorch=True))
print("\nResult of adding 100, back in Python (with original shape):")
print(T3D.compute(asPyTorch=True, shape=T3D_shape))