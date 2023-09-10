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
import MapKernelUtils
import numpy as np
import ctypes
from sympy import symbols, sympify, lambdify, Symbol
import re

def apply_map_function(arg_matrix_ptr, res_matrix_ptr, arg_shape, res_shape, func, varName, dtype):
    
    ctypes_type = MapKernelUtils.get_ctypes_type(dtype)
    arg_buffer = (ctypes_type * (arg_shape[0] * arg_shape[1])).from_address(arg_matrix_ptr)
    res_buffer = (ctypes_type * (res_shape[0] * res_shape[1])).from_address(res_matrix_ptr)
    
    arg_matrix = np.frombuffer(arg_buffer, dtype=MapKernelUtils.get_numpy_type(dtype)).reshape(arg_shape)
    res_matrix = np.frombuffer(res_buffer, dtype=MapKernelUtils.get_numpy_type(dtype)).reshape(res_shape)

    match = re.search(r'def (\w+)', func)
    if match:
        try:
            context = {}
            context[varName] = Symbol(varName)

            exec(func, context)
            func_name = match.groups()[0]
            func_obj = context.get(func_name)
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