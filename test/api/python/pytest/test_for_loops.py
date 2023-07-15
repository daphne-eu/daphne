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
    Y = dctx.fill(0.8, 5, 5)
    def body(x, i):
        more = x + Y
        result  = more + 1
        return result
    output = dctx.for_loop([X], body, 1, 10)
    daphne_ouput = output[0].compute()

    np_X = np.array([3.8 for i in range(25)]).reshape((5,5))
    np_Y = np.array([0.8 for i in range(25)]).reshape((5,5))
    for _ in range(1, 11):
        np_X = np_X + np_Y
        np_X = np_X + 1

    assert np.array_equal(daphne_ouput, np_X)

def test_simple_with_step():
    X = dctx.fill(3.8, 5, 5)
    Y = dctx.fill(0.8, 5, 5)

    def body(x, i):
        more = x + Y
        return more + 1

    output = dctx.for_loop([X], body, 1, 20, 2)
    daphne_ouput = output[0].compute()

    np_X = np.array([3.8 for i in range(25)]).reshape((5,5))
    np_Y = np.array([0.8 for i in range(25)]).reshape((5,5))
    for _ in range(1, 21, 2):
        np_X = np_X + np_Y
        np_X = np_X + 1

    assert np.array_equal(daphne_ouput, np_X)

def test_simple_with_negative_step():
    X = dctx.fill(3.8, 5, 5)

    def body(x, i):
        return x + 1

    output = dctx.for_loop([X], body, 20, 1, -2)
    daphne_ouput = output[0].compute()

    np_X = np.array([3.8 for i in range(25)]).reshape((5,5))
    for _ in range(20, 0, -2):
        np_X = np_X + 1

    assert np.array_equal(daphne_ouput, np_X)

def test_with_itetor_use():
    X = dctx.fill(3.8, 5, 5)

    def body(x, i):
        return x + i

    output = dctx.for_loop([X], body, 1, 10)
    daphne_ouput = output[0].compute()

    np_X = np.array([3.8 for i in range(25)]).reshape((5,5))
    for i in range(1, 11):
        np_X = np_X + i

    assert np.array_equal(daphne_ouput, np_X)


def test_2_inputs():
    X = dctx.fill(3.8, 5, 5)
    Y = dctx.fill(0.8, 5, 5)

    def body(x, y, i):
        more = x + 10
        return more + 1, y + 2

    output = dctx.for_loop([X, Y], body, 1, 10)
    daphne_ouput = (output[0] + output[1]).compute()

    np_X = np.array([3.8 for i in range(25)]).reshape((5,5))
    np_Y = np.array([0.8 for i in range(25)]).reshape((5,5))
    for _ in range(1, 11):
        np_X = np_X + 10
        np_X = np_X + 1
        np_Y = np_Y + 2
    expected_ouput = np_X + np_Y

    assert np.array_equal(daphne_ouput, expected_ouput)

def test_separate_computing():
    X1 = dctx.fill(1.0, 5, 5)
    X2 = dctx.fill(2.0, 5, 5)
    X3 = dctx.fill(3.0, 5, 5)
    X4 = dctx.fill(4.0, 5, 5)
    X5 = dctx.fill(5.0, 5, 5)

    def increment_by_one(x1, x2, x3, x4, x5, i):
        return x1 + 1, x2 + 1, x3 + 1, x4 + 1, x5 + 1

    output = dctx.for_loop([X1, X2, X3, X4, X5], increment_by_one, 1, 10)
    daphne_ouput1 = output[0].compute()
    daphne_ouput2 = output[1].compute()
    daphne_ouput3 = output[2].compute()
    daphne_ouput4 = output[3].compute()
    daphne_ouput5 = output[4].compute()

    np_X1 = np.array([1.0 for i in range(25)]).reshape((5,5))
    np_X2 = np.array([2.0 for i in range(25)]).reshape((5,5))
    np_X3 = np.array([3.0 for i in range(25)]).reshape((5,5))
    np_X4 = np.array([4.0 for i in range(25)]).reshape((5,5))
    np_X5 = np.array([5.0 for i in range(25)]).reshape((5,5))
    for _ in range(1, 11):
        np_X1 = np_X1 + 1
        np_X2 = np_X2 + 1
        np_X3 = np_X3 + 1
        np_X4 = np_X4 + 1
        np_X5 = np_X5 + 1
    
    assert np.array_equal(daphne_ouput1, np_X1)
    assert np.array_equal(daphne_ouput2, np_X2)
    assert np.array_equal(daphne_ouput3, np_X3)
    assert np.array_equal(daphne_ouput4, np_X4)
    assert np.array_equal(daphne_ouput5, np_X5)

def test_invalid_callback_inputs():
    X = dctx.fill(3.8, 5, 5)
    body = lambda x : x + 1
    with pytest.raises(ValueError) as raised_error:
        dctx.for_loop([X], body, 1, 10)

    assert str(raised_error.value) == f"{body} does not have the same number of arguments as input nodes + 1"
    
def test_invalid_callback_outputs():
    X = dctx.fill(3.8, 5, 5)
    body = lambda x, i: (x + 1, i)
    output = dctx.for_loop([X], body, 1, 10)
    with pytest.raises(IndexError):
        output[0].compute()
