import numpy as np
import ctypes
from sympy import symbols, sympify, lambdify
import re

def apply_map_function(arg_matrix_ptr, res_matrix_ptr, arg_shape, res_shape, func, varName, dtype):
    
    ctypes_type = get_ctypes_type(dtype)
    arg_buffer = (ctypes_type * (arg_shape[0] * arg_shape[1])).from_address(arg_matrix_ptr)
    res_buffer = (ctypes_type * (res_shape[0] * res_shape[1])).from_address(res_matrix_ptr)
    
    arg_matrix = np.frombuffer(arg_buffer, dtype=get_type(dtype)).reshape(arg_shape)
    res_matrix = np.frombuffer(res_buffer, dtype=get_type(dtype)).reshape(res_shape)

    match = re.search(r'def (\w+)', func)
    if match:
        try:
            exec(func)
            func_name = match.groups()[0]
            func_obj = locals().get(func_name)
            if func_obj:
                res_matrix[:] = np.vectorize(func_obj)(arg_matrix)
            else:
                print(f"Function '{func_name}' not found.")
        except Exception as e:
            print(f"Failed to execute function: {str(e)}")
    else:
        try:
            x = symbols(varName)
            func_expr = sympify(func.strip())
            func_lambda = lambdify(x, func_expr, modules=["numpy"])
            res_matrix[:] = func_lambda(arg_matrix)
        except Exception as e:
            print(f"Failed to execute lambda expression: {str(e)}")

def get_type(dtype_str):
    """Get the corresponding numpy type for a dtype represented by a string."""
    if dtype_str == "float32":
        return np.float32
    elif dtype_str == "float64":
        return np.double
    elif dtype_str == "int32":
        return np.int32
    elif dtype_str == "int64":
        return np.int64
    elif dtype_str == "int8":
        return np.int8
    elif dtype_str == "uint64":
        return np.uint64
    elif dtype_str == "uint8":
        return np.uint8
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    
def get_ctypes_type(dtype_str):
    """Get the corresponding ctypes type for a dtype represented by a string."""
    dtype_mapping = {
        "float32": ctypes.c_float,
        "float64": ctypes.c_double,
        "int32": ctypes.c_int32,
        "int64": ctypes.c_int64,
        "int8": ctypes.c_int8,
        "uint64": ctypes.c_uint64,
        "uint8": ctypes.c_uint8,
        # Add other type mappings if necessary
    }
    return dtype_mapping.get(dtype_str)