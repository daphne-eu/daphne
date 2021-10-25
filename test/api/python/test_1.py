import os
import unittest
import numpy as np
from api.python.operator.nodes.matrix import Matrix
from api.python.utils.converters import rand
import random

np.random.seed(7)
# TODO Remove the randomness of the test, such that
#      inputs for the random operation is predictable
shape = (random.randrange(1, 25), random.randrange(1, 25))
dist_shape = (10, 15)
min_max = (0, 1)
sparsity = random.uniform(0.0, 1.0)
seed = 123
distributions = ["norm", "uniform"]
dim = 5
m3 = np.array(np.random.randint(100, size=dim * dim) + 1.01, dtype=np.double)
m3.shape = (dim, dim)

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