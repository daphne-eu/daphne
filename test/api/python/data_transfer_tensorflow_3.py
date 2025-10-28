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

# Data transfer from pandas to DAPHNE and back, via shared memory.

import numpy as np
import tensorflow as tf
from daphne.context.daphne_context import DaphneContext

indices = np.array([[0, 2, 1]], dtype=np.int64)
values = np.array([20.0], dtype=np.int64)
shape = (2, 3, 2)

st = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=shape)

dctx = DaphneContext()

A, orig_shape1 = dctx.from_tensorflow(st, shared_memory=True, return_shape=True)
A.print().compute()
