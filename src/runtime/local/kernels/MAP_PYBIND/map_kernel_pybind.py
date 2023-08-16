from sympy import parse_expr, lambdify
import numpy as np

def map_function(arg_matrix, res_matrix, func_str, varName):
   
    input_matrix = np.frombuffer(arg_matrix, dtype=np.double)
    output_matrix = np.frombuffer(res_matrix, dtype=np.double)

    #Easy Eval approach
    #for i in range(len(input_matrix)):
    #    output_matrix[i] = eval(func__str, {'x': input_matrix[i]})

    # Alternatively, you can use the `ctypes` library to work directly with the memory:
    # import ctypes
    # for i in range(len(input_array)):
    #     x = input_array[i]
    #     output_array[i] = eval(func_string)

    # Or you can parse the expression using a library like `sympy` for more complex functions:
    from sympy.parsing.sympy_parser import parse_expr
    func_expr = parse_expr(func_str)
    func = lambdify(varName, func_expr)
    for i in range(len(input_matrix)):
         output_matrix[i] = func(input_matrix[i])