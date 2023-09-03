# -------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# Modifications Copyright 2022 The DAPHNE Consortium
#
# -------------------------------------------------------------

import numpy as np
import ctypes
from sympy import symbols, lambdify, sympify, Symbol
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
    
    match = re.search(r'def (\w+)', func)
    if match:
        try:
            context = {}
            context[varName] = Symbol(varName)

            exec(func, context)
            func_name = match.groups()[0]
            func_obj = context.get(func_name)
            if func_obj:
                res_array[:] = np.vectorize(func_obj)(arg_array)
            else:
                print(f"Function '{func_name}' not found.")
        except Exception as e:
            print(f"Failed to execute function: {str(e)}")
    else:
        try:
            x = symbols(varName)
            func_expr = sympify(func.strip())
            func_lambda = lambdify(x, func_expr, modules=["numpy"])
            res_array[:] = func_lambda(arg_array)
        except Exception as e:
            print(f"Failed to execute lambda expression: {str(e)}")

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