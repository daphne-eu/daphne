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

import ctypes
import numpy as np

#Binary Kernel, Copy kernel, CSV Kernel
def get_numpy_type(dtype_str):
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

#Shared Memory Address Kernel
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