from sympy import parse_expr, lambdify
import numpy as np

def map_function(arg_matrix, res_matrix, func_str, varName):
    input_matrix = np.array(arg_matrix, dtype=np.double)
    output_matrix = np.array(res_matrix, dtype=np.double)

    func_expr = parse_expr(func_str)
    func = lambdify(varName, func_expr)

    for i in range(len(input_matrix)):
        output_matrix[i] = func(input_matrix[i])

    return output_matrix