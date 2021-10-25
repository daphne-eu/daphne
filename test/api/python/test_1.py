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
from api.python.operator.nodes.matrix import Matrix
from api.python.utils.converters import rand
import random

np.random.seed(7)
# TODO Remove the randomness of the test, such that
#      inputs for the random operation is predictable
shape = (random.randrange(1, 7), random.randrange(1, 7))
dist_shape = (10, 15)
min_max = (0, 1)
sparsity = random.uniform(0.0, 1.0)
seed = 123


m1 = rand(rows=shape[0], cols=shape[1],
                          min=min_max[0], max=min_max[1], seed=seed, sparsity=sparsity)

m2 = rand(rows=shape[0], cols=shape[1],
                          min=min_max[0], max=min_max[1], seed=seed, sparsity=sparsity)

class TestBinaryOp(unittest.TestCase):
    def test_binary_op(self):
        print("Mat1:    ")
        m1.print().compute()
        print("Mat2:    ")
        m2.print().compute()
        print("M1 + M2:")
        (m1+m2).compute()
        print("M1 - M2:")
        (m1-m2).compute()

    def test_sum1(self):
        print("Sum of M1:")
        m1.sum().compute()

if __name__ == "__main__":
    unittest.main(exit=False)