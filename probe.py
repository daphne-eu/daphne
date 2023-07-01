import sys
sys.path.append("/home/ubuntu/daphne/src")

import numpy as np
from api.python.context.daphne_context import DaphneContext

dctx = DaphneContext()

def test_while1():
    X = dctx.fill(3.8, 5, 5)
    Y = dctx.fill(0.8, 5, 5)

    def while_body(x):
        result = x - Y #dctx.while_loop(X, lambda x: x > 0, lambda x: x - X)
        return result

    output = dctx.while_loop([X], lambda x: x.sum() > 0, while_body)
    print(output[0].compute())

def test_while2():
    X = dctx.fill(3.8, 5, 5)
    Y = dctx.fill(2.8, 5, 5)
    Z = dctx.fill(0.8, 5, 5)

    def while_body(x, y):
        return x - Z, y - Z
    
    def pred(x, y):
        return dctx.logical_or(x.sum() > 0, y.sum() > 0)

    output = dctx.while_loop([X, Y], pred, while_body)
    print((output[0] + output[0]).compute())

def test_if_else1():
    X = dctx.fill(3.8, 5, 5)
    Y = dctx.fill(0.8, 5, 5)
    Z = dctx.fill(1.8, 5, 5)

    def true_fn(x): 
        return x - Z
    
    def false_fn(x):
        R = dctx.fill(2.8, 5, 5)
        return x + Z,
    
    def pred():
        return X.sum() < 10
    
    cond_statement = dctx.if_else([X], lambda: X.sum() < 10, true_fn, false_fn)
    return print((cond_statement[0]).compute())

def test_if_else2():
    X = dctx.fill(3.8, 5, 5)
    Y = dctx.fill(0.8, 5, 5)
    Z = dctx.fill(1.8, 5, 5)

    def true_fn(x): 
        return x - Z
    
    def false_fn(x):
        R = dctx.fill(2.8, 5, 5)
        return x + Z,
    
    def pred():
        return dctx.logical_and(X.sum() < 10, X.sum() > 200)
    
    cond_statement = dctx.if_else([X], pred, true_fn, false_fn)
    return print((cond_statement[0]).compute())

def test_for_loop1():
    X = dctx.fill(3.8, 5, 5)
    Y = dctx.fill(0.8, 5, 5)

    def body(x, i):
        more = x + 10
        return more + 1,

    output = dctx.for_loop([X], body, 1, 10, 2)
    print((output[0]).compute())


def test_for_loop2():
    X = dctx.fill(3.8, 5, 5)
    Y = dctx.fill(0.8, 5, 5)

    def body(x, y, i):
        more = x + 10
        return more + 1, y + 2

    output = dctx.for_loop([X, Y], body, 1, 10, 2)
    print((output[0] + output[1]).compute())

def test_user_def_func():
    X = dctx.fill(3.8, 5, 5)

    @dctx.function
    def increment_by_one(x):
        inter = x + 10
        cond = dctx.if_else([x], lambda: inter.sum() > 0, lambda x: x + 5, lambda x: x - 5)
        return x, x + 1, inter, cond[0]
    
    print((increment_by_one(X)[0] + increment_by_one(X)[0] + increment_by_one(X)[3]).compute())


if __name__ == "__main__":
    test_for_loop1()
    test_for_loop2()
    test_if_else1()
    test_if_else2()
    test_while1()
    test_while2()
    test_user_def_func()
