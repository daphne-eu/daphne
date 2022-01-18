# -------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# -------------------------------------------------------------
import unittest
import numpy as np
from api.python.utils.converters import from_numpy

import random

np.random.seed(7)
# TODO Remove the randomness of the test, such that
#      inputs for the random operation is predictable
shape = (random.randrange(1, 7), random.randrange(1, 7))
dist_shape = (10, 15)
min_max = (0, 1)
sparsity = random.uniform(0.0, 1.0)
seed = 123
dim = 5
np.random.seed(7)
m3 = np.array(np.random.randint(100, size=dim*dim)+1.01, dtype=np.double)
m3.shape = (dim, dim)
m4 = np.array(np.random.randint(5, size=dim*dim)+1, dtype=np.double)
m4.shape = (dim,dim)


class TestBinaryOp(unittest.TestCase):
   
    
    def test_plus(self):
        result = ((from_numpy(m3)*from_numpy(m4)).compute())
        self.assertTrue(np.allclose(result, m3*m4))

    def test_mult(self):
        result = ((from_numpy(m3)*from_numpy(m4)).compute())
        self.assertTrue(np.allclose(result, m3*m4))

    def test_div(self):
        result = ((from_numpy(m3)*from_numpy(m4)).compute())
        self.assertTrue(np.allclose(result, m3*m4))
        
if __name__ == "__main__":
    unittest.main(exit=False)
