import numpy as np
import ctypes
#from sympy import symbols, lambdify
import re

def apply_map_function(upper_res, lower_res, upper_arg, lower_arg, rows, cols, func, varName, dtype_arg, dtype_res):
    res_ptr = ((upper_res << 32) | lower_res)
    arg_ptr = ((upper_arg << 32) | lower_arg)

    res_array = np.ctypeslib.as_array(
        ctypes.cast(res_ptr, ctypes.POINTER(get_ctypes_type(dtype_res))),
        shape=(rows, cols)
    )
    arg_array = np.ctypeslib.as_array(
        ctypes.cast(arg_ptr, ctypes.POINTER(get_ctypes_type(dtype_arg))),
        shape=(rows, cols)
    )
    
    #Eval Approach
    #res_array[:] = eval(func.replace(varName, 'arg_array'))

    #Lambdify Approach
    #x = symbols(varName)
    #func = lambdify(x, func, modules=["numpy"])
    #res_array[:] = func(arg_array)

    #Exec and Re Approach for full functions defined
    match = re.search(r'def (\w+)', func)
    if match:
        try:
            exec(func)
            func_name = match.groups()[0]
            func_obj = locals().get(func_name)
            if func_obj:
                res_array[:] = np.vectorize(func_obj)(arg_array)
            else:
                print(f"Function '{func_name}' not found.")
        except Exception as e:
            print(f"Failed to execute function: {str(e)}")
    else:
        print("No function name found")


def get_ctypes_type(dtype_str):
    """Get the corresponding ctypes type for a dtype represented by a string."""
    if dtype_str == "float32":
        return ctypes.c_float
    elif dtype_str == "float64":
        return ctypes.c_double
    elif dtype_str == "int32":
        return ctypes.c_int32
    elif dtype_str == "int64":
        return ctypes.c_int64
    elif dtype_str == "int8":
        return ctypes.c_int8
    elif dtype_str == "uint64":
        return ctypes.c_uint64
    elif dtype_str == "uint8":
        return ctypes.c_uint8
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")