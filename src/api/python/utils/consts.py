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
# -------------------------------------------------------------
from typing import Union
import os
import ctypes 

VALID_INPUT_TYPES = Union['DAGNode', str, int, float, bool]
BINARY_OPERATIONS = ['+', '-', '/', '*', '<', '<=', '>', '>=', '==', '!=', '@']
VALID_ARITHMETIC_TYPES = Union['DAGNode', int, float]

PYTHON_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

TMP_PATH = os.path.join(PYTHON_PATH, "tmp")

PROTOTYPE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(PYTHON_PATH)))

libDaphneShared = ctypes.CDLL(PROTOTYPE_PATH+"/build/src/api/shared/libDaphneShared.so")
       

#Values need to be updated accordingly to the valuetypecodes in ValueTypeCode.h, as this is 1/1 copy 
SI8 = 0
SI32 = 1
SI64 = 2
UI8 = 3
UI32 = 4
UI64 = 5
F32 = 6
F64 = 7

#Python represantation of the struct DaphneLibResult
class DaphneLibResult(ctypes.Structure):
    _fields_ = [("address", ctypes.c_void_p), ("rows", ctypes.c_int64), ("cols", ctypes.c_int64), ("vtc", ctypes.c_int)]
    