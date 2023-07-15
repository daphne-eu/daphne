# ----------------------------------------
# This test is to be executed from inside
# the main DAPHNE folder with 'pytest'.
# ----------------------------------------

# Prepare the environment
#########################
import os
import sys
CUURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DAPHNE_SRC_DIR = os.path.abspath(os.path.join(CUURRENT_DIR, '..', '..', '..', '..', 'src'))
sys.path.append(DAPHNE_SRC_DIR)
#########################

import pytest
import numpy as np
from api.python.context.daphne_context import DaphneContext

dctx = DaphneContext()

def test_simple():
    X = dctx.fill(3.8, 5, 5)

    @dctx.function
    def increment(x):
        return x + 1
    
    output = increment(X)
    daphne_output = output[0].compute()

    np_X = np.array([3.8 for i in range(25)]).reshape((5,5))
    expected_output = np_X + 1

    assert np.array_equal(daphne_output, expected_output)

def test_1_input_3_outputs():
    X = dctx.fill(3.8, 5, 5)

    @dctx.function
    def increment(x):
        return x, x + 1, x + 2
    
    output = increment(X)
    daphne_output = (output[0] + output[1] + output[2]).compute()

    np_X = np.array([3.8 for i in range(25)]).reshape((5,5))
    expected_output = np_X + (np_X + 1) + (np_X + 2)

    assert np.array_equal(daphne_output, expected_output)

def test_with_for_loop():
    X = dctx.fill(3.8, 5, 5)

    @dctx.function
    def use_for_loop(x):
        return dctx.for_loop([x], lambda node, i: node + i, 1, 10)
    
    output = use_for_loop(X)
    daphne_output = output[0].compute()

    np_X = np.array([3.8 for i in range(25)]).reshape((5,5))
    for i in range(1,11):
        np_X = np_X + i

    assert np.array_equal(daphne_output, np_X)

def test_with_while_loop():
    X = dctx.fill(3.8, 5, 5)

    @dctx.function
    def use_while_loop(x):
        return dctx.while_loop([x], lambda node: node.sum() > 0.0, lambda node: node - 1)
    
    output = use_while_loop(X)
    daphne_output = output[0].compute()

    np_X = np.array([3.8 for i in range(25)]).reshape((5,5))
    while np_X.sum() > 0.0:
        np_X = np_X - 1

    assert np.array_equal(daphne_output, np_X)

@pytest.mark.parametrize("num", [0.1, 3.8])
def test_with_cond(num):
    X = dctx.fill(num, 5, 5)

    @dctx.function
    def use_cond(x):
        return dctx.cond([x], lambda: X.sum() < 10.0, lambda node: node - 1, lambda node: node + 1)
    
    output = use_cond(X)
    daphne_output = output[0].compute()

    np_X = np.array([num for i in range(25)]).reshape((5,5))
    if np_X.sum() < 10.0:
        np_X = np_X - 1
    else:
        np_X = np_X + 1

    assert np.array_equal(daphne_output, np_X)

def test_complex():
    X = dctx.fill(3.8, 5, 5)
    Y = dctx.fill(0.8, 5, 5)

    @dctx.function
    def do_several_things(x):
        x = x @ Y
        x = x + 1
        x = x.sum(1)
        return x
    
    output = do_several_things(X)
    daphne_output = output[0].compute()

    np_X = np.array([3.8 for i in range(25)]).reshape((5,5))
    np_Y = np.array([0.8 for i in range(25)]).reshape((5,5))
    np_X = np_X @ np_Y
    np_X = np_X + 1
    np_X = np_X.sum(axis=1)
    # reshape the matrix resulted from the numpy calculation since the output is in different dimension
    assert np.array_equal(daphne_output, np_X.reshape(1, -1))

def test_separate_computing():
    X1 = dctx.fill(0.8, 5, 5)
    X2 = dctx.fill(1.8, 5, 5)
    X3 = dctx.fill(2.8, 5, 5)

    @dctx.function
    def increment_multiple(x1, x2, x3):
        return x1 + 1, x2 + 1, x3 + 1
    
    output = increment_multiple(X1, X2, X3)
    daphne_output1 = output[0].compute()
    daphne_output2 = output[1].compute()
    daphne_output3 = output[2].compute()

    np_X1 = np.array([0.8 for i in range(25)]).reshape((5,5))
    np_X2 = np.array([1.8 for i in range(25)]).reshape((5,5))
    np_X3 = np.array([2.8 for i in range(25)]).reshape((5,5))
    np_X1 = np_X1 + 1
    np_X2 = np_X2 + 1
    np_X3 = np_X3 + 1
    # reshape the matrix resulted from the numpy calculation since the output is in different dimension
    assert np.array_equal(daphne_output1, np_X1)
    assert np.array_equal(daphne_output2, np_X2)
    assert np.array_equal(daphne_output3, np_X3)
