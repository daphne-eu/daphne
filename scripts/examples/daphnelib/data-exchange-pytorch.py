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
import torch
import numpy as np

dc = DaphneContext()

# Example usage for a 3x3 tensor
tensor2d = torch.tensor(np.random.randn(4, 3))

# Print the tensor
print("How the 2d Tensor looks in Python:")
print(tensor2d)

T2D = dc.from_pytorch(tensor2d)

print("\nHow DAPHNE sees the 2d tensor from PyTorch:")
print(T2D.compute(isPytorch=True))

# Example usage for a 3x3x3 tensor
tensor3d = torch.tensor(np.random.randn(4, 3, 4))

# Print the tensor
print("\nHow the 3d Tensor looks in Python:")
print(tensor3d)

T3D, T3D_shape = dc.from_pytorch(tensor3d, return_shape=True)

print("\nHow DAPHNE sees the 3d tensor from PyTorch:")
print(T3D.compute(isPytorch=True))

print("\nHow the original shape of the tensor looks like:")
print(T3D_shape)

print("\nHow the 3d tensor looks transformed back to the original shape:")
tensor3d_back = T3D.compute(isPytorch=True, shape=T3D_shape)
print(tensor3d_back)

# Example usage for a 4x3x3x3 tensor
tensor4d = torch.tensor(np.random.randn(3, 3, 3, 3))

# Print the tensor
print("\nHow the 4d Tensor looks in Python:")
print(tensor4d)

T4D, T4D_shape = dc.from_pytorch(tensor4d, verbose=True, return_shape=True)

print("\nHow DAPHNE sees the 4d tensor from PyTorch:")
print(T4D.compute(verbose=True, isPytorch=True))

print("\nHow the original shape of the tensor looks like:")
print(T4D_shape)

tensor4d_back = T4D.compute(verbose=True, isPytorch=True, shape=T4D_shape)
print("\nHow the 4d tensor looks transformed back to the original shape:")
print(tensor4d_back)