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

@pytest.mark.parametrize("num", [0.1, 3.8])
def test_simple(num):
    X = dctx.fill(num, 5, 5)

    def true_fn(x): 
        return x - 1
    
    def false_fn(x):
        return x + 1,
    
    cond_statement = dctx.cond([X], lambda: X.sum() < 10, true_fn, false_fn)
    daphne_output = cond_statement[0].compute()

    np_X = np.array([num for i in range(25)]).reshape((5,5))
    if np_X.sum() < 10:
        np_X = np_X - 1
    else:
        np_X = np_X + 1
    
    assert np.array_equal(np_X, daphne_output)

@pytest.mark.parametrize("num", [0.1, 3.8])
def test_simple_with_outer_scope_matrix(num):
    X = dctx.fill(num, 5, 5)
    Y = dctx.fill(1.8, 5, 5)

    def true_fn(x): 
        return x - Y
    
    def false_fn(x):
        R = dctx.fill(2.8, 5, 5)
        return x + Y,
    
    cond_statement = dctx.cond([X], lambda: X.sum() < 10, true_fn, false_fn)
    daphne_output = cond_statement[0].compute()

    np_X = np.array([num for i in range(25)]).reshape((5,5))
    np_Y = np.array([1.8 for i in range(25)]).reshape((5,5))
    if np_X.sum() < 10:
        np_X = np_X - np_Y
    else:
        np_X = np_X + np_Y
    
    assert np.array_equal(np_X, daphne_output)

@pytest.mark.parametrize("num", [0.1, 3.8])
def test_simple_with_inner_scope_matrix(num):
    X = dctx.fill(num, 5, 5)

    def true_fn(x): 
        Y = dctx.fill(1.8, 5, 5)
        return x - Y
    
    def false_fn(x):
        R = dctx.fill(2.8, 5, 5)
        return x + R,
    
    cond_statement = dctx.cond([X], lambda: X.sum() < 10, true_fn, false_fn)
    daphne_output = cond_statement[0].compute()

    np_X = np.array([num for i in range(25)]).reshape((5,5))
    np_Y = np.array([1.8 for i in range(25)]).reshape((5,5))
    np_R = np.array([2.8 for i in range(25)]).reshape((5,5))
    if np_X.sum() < 10:
        np_X = np_X - np_Y
    else:
        np_X = np_X + np_R
    
    assert np.array_equal(np_X, daphne_output)

@pytest.mark.parametrize("num", [3.8, 10.0])
def test_complex_pred(num):
    X = dctx.fill(num, 5, 5)

    def true_fn(x): 
        return x - 1
    
    def false_fn(x):
        return x + 1
    
    def pred():
        return dctx.logical_and(X.sum() < 10, X.sum() > 200)
    
    cond_statement = dctx.cond([X], pred, true_fn, false_fn)
    daphne_output = cond_statement[0].compute()

    np_X = np.array([num for i in range(25)]).reshape((5,5))
    if np_X.sum() < 10 and np_X.sum() > 200:
        np_X = np_X - 1
    else:
        np_X = np_X + 1
    
    assert np.array_equal(np_X, daphne_output)

@pytest.mark.parametrize("num", [0.1, 3.8])
def test_2_outputs(num):
    X = dctx.fill(num, 5, 5)
    Y = dctx.fill(1.8, 5, 5)

    def true_fn(x, y): 
        return x - 1, y + 1
    
    def false_fn(x, y):
        return x + 1, y - 1
    
    cond_statement = dctx.cond([X, Y], lambda: X.sum() < 10, true_fn, false_fn)
    daphne_output = (cond_statement[0] + cond_statement[1]).compute()

    np_X = np.array([num for i in range(25)]).reshape((5,5))
    np_Y = np.array([1.8 for i in range(25)]).reshape((5,5))
    if np_X.sum() < 10:
        np_X = np_X - 1
        np_Y = np_Y + 1
    else:
        np_X = np_X + 1
        np_Y = np_Y - 1
    expected_output = np_X + np_Y

    assert np.array_equal(daphne_output, expected_output)

@pytest.mark.parametrize("num", [0.1, 3.8])
def test_2_outputs_complex_cond(num):
    X = dctx.fill(num, 5, 5)
    Y = dctx.fill(1.8, 5, 5)

    def true_fn(x, y): 
        return x - 1, y + 1
    
    def false_fn(x, y):
        return x + 1, y - 1

    def pred():
        return dctx.logical_or(X.sum() < 10, Y.sum() > 200.0)
    
    cond_statement = dctx.cond([X, Y], pred, true_fn, false_fn)
    daphne_output = (cond_statement[0] + cond_statement[1]).compute()

    np_X = np.array([num for i in range(25)]).reshape((5,5))
    np_Y = np.array([1.8 for i in range(25)]).reshape((5,5))
    if np_X.sum() < 10:
        np_X = np_X - 1
        np_Y = np_Y + 1
    else:
        np_X = np_X + 1
        np_Y = np_Y - 1
    expected_output = np_X + np_Y

    assert np.array_equal(daphne_output, expected_output)

@pytest.mark.parametrize("index", [0, 4])
def test_separate_computing(index):
    X1 = dctx.fill(1.0, 5, 5)
    X2 = dctx.fill(2.0, 5, 5)
    X3 = dctx.fill(3.0, 5, 5)
    X4 = dctx.fill(4.0, 5, 5)
    X5 = dctx.fill(5.0, 5, 5)
    nodes = [X1, X2, X3, X4, X5]
    def pred():
        return nodes[index].sum() < 100.0

    def increment_by_one(x1, x2, x3, x4, x5):
        return x1 + 1, x2 + 1, x3 + 1, x4 + 1, x5 + 1
    
    def do_nothing(x1, x2, x3, x4, x5):
        return x1, x2, x3, x4, x5

    output = dctx.cond([X1, X2, X3, X4, X5], pred, increment_by_one, do_nothing)
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
    arrays = [np_X1, np_X2, np_X3, np_X4, np_X5]
    if arrays[index].sum() < 100.0:
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
