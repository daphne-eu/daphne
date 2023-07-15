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

    def do_while_body(x):
        result = x - 1
        return result

    output = dctx.do_while_loop([X], lambda x: x.sum() > 0, do_while_body)
    daphne_output = output[0].compute()

    expected_output = np.array([3.8 for i in range(25)]).reshape((5,5))
    expected_output = expected_output - 1
    while expected_output.sum() > 0:
        expected_output = expected_output - 1

    assert np.array_equal(daphne_output, expected_output)

def test_simple_outer_scope_var():
    X = dctx.fill(3.8, 5, 5)
    Y = dctx.fill(0.8, 5, 5)

    def do_while_body(x):
        result = x - Y
        return result

    output = dctx.do_while_loop([X], lambda x: x.sum() > 0, do_while_body)
    daphne_output = output[0].compute()

    expected_output = np.array([3.8 for i in range(25)]).reshape((5,5))
    np_Y = np.array([0.8 for i in range(25)]).reshape((5,5))
    expected_output = expected_output - np_Y
    while expected_output.sum() > 0:
        expected_output = expected_output - np_Y

    assert np.array_equal(daphne_output, expected_output)

def test_complex_condtion():
    X = dctx.fill(9.9, 5, 5)

    def do_while_body(x):
        result = x - 1
        return result
    
    def cond(x):
        return dctx.logical_and(x.sum() > 1.0, x.aggMax() > 1.0)

    output = dctx.do_while_loop([X], cond, do_while_body)
    daphne_output = output[0].compute()

    expected_output = np.array([9.9 for i in range(25)]).reshape((5,5))
    expected_output = expected_output - 1
    while expected_output.sum() > 1.0 and expected_output.max() > 1.0:
        expected_output = expected_output - 1

    assert np.array_equal(daphne_output, expected_output)

def test_2_inputs():
    X = dctx.fill(3.8, 5, 5)
    Y = dctx.fill(2.8, 5, 5)
    Z = dctx.fill(0.8, 5, 5)

    def do_while_body(x, y):
        return x - Z, y - Z
    
    def cond(x, y):
        return x.sum() > 0

    output = dctx.do_while_loop([X, Y], cond, do_while_body)
    daphne_output = (output[0] + output[1]).compute()

    np_X = np.array([3.8 for i in range(25)]).reshape((5,5))
    np_Y = np.array([2.8 for i in range(25)]).reshape((5,5))
    np_Z = np.array([0.8 for i in range(25)]).reshape((5,5))
    
    np_X = np_X - np_Z
    np_Y = np_Y - np_Z
    while np_X.sum() > 0:
        np_X = np_X - np_Z
        np_Y = np_Y - np_Z
    expected_output = np_X + np_Y

    assert np.array_equal(daphne_output, expected_output)

def test_2_inputs_complex_condition():
    X = dctx.fill(3.8, 5, 5)
    Y = dctx.fill(2.8, 5, 5)
    Z = dctx.fill(0.8, 5, 5)

    def do_while_body(x, y):
        return x - Z, y - Z
    
    def cond(x, y):
        return dctx.logical_or(x.sum() > 0, y.sum() > 0)

    output = dctx.do_while_loop([X, Y], cond, do_while_body)
    daphne_output = (output[0] + output[1]).compute()

    np_X = np.array([3.8 for i in range(25)]).reshape((5,5))
    np_Y = np.array([2.8 for i in range(25)]).reshape((5,5))
    np_Z = np.array([0.8 for i in range(25)]).reshape((5,5))
    
    np_X = np_X - np_Z
    np_Y = np_Y - np_Z
    while np_X.sum() > 0 or np_Y.sum() > 0:
        np_X = np_X - np_Z
        np_Y = np_Y - np_Z
    expected_output = np_X + np_Y

    assert np.array_equal(daphne_output, expected_output)

def test_separate_computing():
    X1 = dctx.fill(1.0, 5, 5)
    X2 = dctx.fill(2.0, 5, 5)
    X3 = dctx.fill(3.0, 5, 5)
    X4 = dctx.fill(4.0, 5, 5)
    X5 = dctx.fill(5.0, 5, 5)

    def cond(x1, x2, x3, x4, x5):
        return x1.sum() < 100.0

    def increment_by_one(x1, x2, x3, x4, x5):
        return x1 + 1, x2 + 1, x3 + 1, x4 + 1, x5 + 1

    output = dctx.do_while_loop([X1, X2, X3, X4, X5], cond, increment_by_one)
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
    
    np_X1 = np_X1 + 1
    np_X2 = np_X2 + 1
    np_X3 = np_X3 + 1
    np_X4 = np_X4 + 1
    np_X5 = np_X5 + 1
    while np_X1.sum() < 100.0:
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
    cond = lambda x, i: x.sum() > 0
    do_while_body = lambda x, i: x - 1

    with pytest.raises(ValueError) as raised_error:
        dctx.do_while_loop([X], cond, do_while_body)
    
    assert str(raised_error.value) == f"{do_while_body} does not have the same number of arguments as input nodes"
    

def test_invalid_number_outputs():
    X = dctx.fill(3.8, 5, 5)
    Y = dctx.fill(3.8, 5, 5)
    cond = lambda x, y: x.sum() > 0
    do_while_body = lambda x, y: x - 1
    output = dctx.do_while_loop([X, Y], cond, do_while_body)

    with pytest.raises(IndexError):
        output[2].compute()

def test_invalid_cond_inputs():
    X = dctx.fill(3.8, 5, 5)
    cond = lambda: X.sum() > 0
    do_while_body = lambda x: x - 1

    with pytest.raises(ValueError) as raised_error:
        dctx.do_while_loop([X], cond, do_while_body)
    
    assert str(raised_error.value) == f"{cond} and {do_while_body} do not have the same number of arguments"

if __name__ == "__main__":
    test_simple()
    #test_simple_outer_scope_var()