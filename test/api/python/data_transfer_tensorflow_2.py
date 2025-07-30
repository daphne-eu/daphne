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

indices1 = np.array([[2, 0], [2, 2]], dtype=np.int64)
values1 = np.array([3.84985, 1.15568], dtype=np.float64)
shape1 = (3, 3)
st1 = tf.sparse.SparseTensor(indices=indices1, values=values1, dense_shape=shape1)

dctx = DaphneContext()

A = dctx.from_tensorflow(st1, shared_memory=True)
A.print().compute()