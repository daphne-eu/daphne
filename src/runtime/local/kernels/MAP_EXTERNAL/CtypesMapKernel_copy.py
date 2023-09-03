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
from sympy import symbols, lambdify, sympify, Symbol
import re

def apply_map_function(arg_list, rows, cols, func, varName, dtype_arg):
    arg_array = np.array(arg_list, dtype=get_type(dtype_arg)).reshape(rows, cols)
    
    match = re.search(r'def (\w+)', func)
    if match:
        try:
            context = {}
            context[varName] = Symbol(varName)

            exec(func, context)
            func_name = match.groups()[0]
            func_obj = context.get(func_name)
            if func_obj:
                res_array = np.vectorize(func_obj, otypes=[get_type(dtype_arg)])(arg_array)
                return res_array.flatten().tolist()
            else:
                print(f"Function '{func_name}' not found.")
        except Exception as e:
            print(f"Failed to execute function: {str(e)}")
    else:
        try:
            x = symbols(varName)
            func_expr = sympify(func.strip())
            func_lambda = lambdify(x, func_expr, modules=["numpy"])
            res_array = np.array(func_lambda(arg_array), dtype=get_type(dtype_arg))
            return res_array.flatten().tolist()
        except Exception as e:
            print(f"Failed to execute lambda expression: {str(e)}")
    return []  # Return an empty list if there's an error


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