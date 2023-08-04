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
import tensorflow as tf
import numpy as np

dc = DaphneContext()

# Example usage for a 3x3 tensor
tensor2d = tf.constant(np.random.randn(4,3))

# Print the tensor
print("How the 2d Tensor looks in Python:")
print(tensor2d)

T2D = dc.from_tensorflow(tensor2d)

print("\nHow DAPHNE sees the 2d tensor from tensorflow:")
print(T2D.compute(isTensorflow=True))


# Example usage for a 3x3x3 tensor
tensor3d = tf.constant(np.random.randn(4,3,4))

# Print the tensor
print("\nHow the 3d Tensor looks in Python:")
print(tensor3d)

T3D, T3D_shape = dc.from_tensorflow(tensor3d, return_shape=True)

print("\nHow DAPHNE sees the 3d tensor from tensorflow:")
print(T3D.compute(isTensorflow=True))

print("\nHow the original shape of the tensor looks like:")
print(T3D_shape)

print("\nHow the 3d tensor looks transformed back to the original shape:")
tensor3d_back = T3D.compute(isTensorflow=True, shape=T3D_shape)
print(tensor3d_back)

# Example usage for a 4x3x3x3 tensor
tensor4d = tf.constant(np.random.randn(3,3,3,3))

# Print the tensor
print("\nHow the 4d Tensor looks in Python:")
print(tensor4d)

T4D, T4D_shape = dc.from_tensorflow(tensor4d, verbose=True, return_shape=True)

print("\nHow DAPHNE sees the 4d tensor from tensorflow:")
print(T3D.compute(verbose=True, isTensorflow=True))

print("\nHow the original shape of the tensor looks like:")
print(T3D_shape)

tensor4d_back = T4D.compute(verbose=True, isTensorflow=True, shape=T4D_shape)
print("\nHow the 4d tensor looks transformed back to the original shape:")
print(tensor4d_back)