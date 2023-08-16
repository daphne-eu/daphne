from sympy import parse_expr, lambdify
import numpy as np

def map_function(arg_matrix_ptr, res_matrix_ptr, arg_shape, res_shape, func_str, varName, dtype):
    arg_matrix = np.frombuffer(arg_matrix_ptr, dtype=dtype).reshape(arg_shape)
    res_matrix = np.frombuffer(res_matrix_ptr, dtype=dtype).reshape(res_shape)

    func_expr = parse_expr(func_str)
    func = lambdify(varName, func_expr)

    for i in range(len(arg_matrix)):
        res_matrix[i] = func(arg_matrix[i])