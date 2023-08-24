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
import copy
#########################

import pytest
import numpy as np
from api.python.context.daphne_context import DaphneContext

@pytest.fixture
def dctx():
    return DaphneContext()

def test_simple(dctx):
    X = dctx.fill(3.8, 5, 5)

    @dctx.function
    def increment(x):
        return x + 1
    
    output = increment(X)
    daphne_output = output[0].compute()

    np_X = np.array([3.8 for i in range(25)]).reshape((5,5))
    expected_output = np_X + 1

    assert np.array_equal(daphne_output, expected_output)

def test_1_input_3_outputs(dctx):
    X = dctx.fill(3.8, 5, 5)
    @dctx.function
    def increment(x):
        return x, x + 1, x + 2
    output = increment(X)
    
    np_X = np.array([3.8 for i in range(25)]).reshape((5,5))

    # test first return value
    daphne_output = output[0].compute()
    expected_output = np_X
    assert np.array_equal(daphne_output, expected_output)
    # test second return value
    daphne_output = output[1].compute()
    expected_output = np_X + 1
    assert np.array_equal(daphne_output, expected_output)
    # test third return value
    daphne_output = output[2].compute()
    expected_output = np_X + 2
    assert np.array_equal(daphne_output, expected_output)

def test_multiple_functions(dctx):
    X = dctx.fill(3.8, 5, 5)

    @dctx.function
    def increment(x):
        return x + 2
    
    @dctx.function
    def decrement(x):
        return x - 1
    
    @dctx.function
    def add(x, y):
        return x + y

    np_X = np.array([3.8 for i in range(25)]).reshape((5,5))

    output = increment(X)
    daphne_output = output[0].compute()
    expected_output = np_X + 2
    assert np.array_equal(daphne_output, expected_output)

    output = decrement(output[0])
    daphne_output = output[0].compute()
    expected_output = expected_output - 1
    assert np.array_equal(daphne_output, expected_output)

    output = add(X, output[0])
    daphne_output = output[0].compute()
    expected_output = np_X + expected_output
    assert np.array_equal(daphne_output, expected_output)

def test_multiple_calls(dctx):
    X = dctx.fill(3.8, 5, 5)

    @dctx.function
    def increment(x):
        return x + 1

    np_X = np.array([3.8 for i in range(25)]).reshape((5,5))

    output = increment(X)
    daphne_output = output[0].compute()
    expected_output = np_X + 1
    assert np.array_equal(daphne_output, expected_output)

    output = increment(output[0])
    daphne_output = output[0].compute()
    expected_output = expected_output + 1
    assert np.array_equal(daphne_output, expected_output)

    output = increment(output[0])
    daphne_output = output[0].compute()
    expected_output = expected_output + 1
    assert np.array_equal(daphne_output, expected_output)

@pytest.mark.skip("Unclear DAPHNE function behaviour")
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

@pytest.mark.skip("Unclear DAPHNE function behaviour")
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

@pytest.mark.skip("Unclear DAPHNE function behaviour")
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

@pytest.mark.skip("Unclear DAPHNE function behaviour")
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
    np_X = np_X.sum(axis=0, keepdims=True)
    assert np.allclose(daphne_output, np_X)

