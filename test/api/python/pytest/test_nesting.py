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

@pytest.mark.parametrize("num", [3.8, 10.0])
def test_nested_for_loop_in_cond(num):
    X = dctx.fill(num, 5, 5)

    def true_fn(x): 
        return dctx.for_loop([x], lambda n, i: n - 1, 1, 10)
    
    def false_fn(x):
        return dctx.for_loop([x], lambda n, i: n + 1, 1, 10)
    
    def pred():
        return X.sum() < 10
    
    cond_statement = dctx.cond([X], pred, true_fn, false_fn)
    daphne_output = cond_statement[0].compute()

    np_X = np.array([num for i in range(25)]).reshape((5,5))
    if np_X.sum() < 10 and np_X.sum() > 200:
        for i in range(1, 11):
            np_X = np_X - 1
    else:
        for i in range(1, 11):
            np_X = np_X + 1
    
    assert np.array_equal(np_X, daphne_output)

def test_nested_cond_in_forloop():
    X = dctx.fill(3.8, 5, 5)

    def true_fn(x): 
        return x - 1
    
    def false_fn(x):
        return x + 1
    
    def pred():
        return X.sum() < 10

    def for_body(x, i):
        return dctx.cond([x], pred, true_fn, false_fn)
    
    for_loop = dctx.for_loop([X], for_body, 1, 10)

    daphne_output = for_loop[0].compute()

    np_X = np.array([3.8 for i in range(25)]).reshape((5,5))
    for i in range(1, 11):
        if np_X.sum() < 10:
            np_X = np_X - 1
        else:
            np_X = np_X + 1
    
    assert np.array_equal(np_X, daphne_output)

def test_nested_for_loop_in_for_loop():
    X = dctx.fill(3.8, 5, 5)

    def for_loop_body_nested(x, i):
        return x + 1

    def for_body(x, i):
        return dctx.for_loop([x], for_loop_body_nested, 1, 5)
    
    for_loop = dctx.for_loop([X], for_body, 1, 10)

    daphne_output = for_loop[0].compute()

    np_X = np.array([3.8 for i in range(25)]).reshape((5,5))
    for i in range(1, 11):
        for j in range(1, 6):
            np_X += 1
    
    assert np.array_equal(np_X, daphne_output)
