# Copyright 2023 The DAPHNE Consortium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ctypes
import os

from daphne.utils.consts import PROTOTYPE_PATH, DAPHNELIB_FILENAME

# Python representation of the struct DaphneLibResult.
class DaphneLibResult(ctypes.Structure):
    _fields_ = [
        # For matrices.
        ("address", ctypes.c_void_p),
        ("rows", ctypes.c_int64),
        ("cols", ctypes.c_int64),
        ("vtc", ctypes.c_int64),
        # For frames.
        ("vtcs", ctypes.POINTER(ctypes.c_int64)),
        ("labels", ctypes.POINTER(ctypes.c_char_p)),
        ("columns", ctypes.POINTER(ctypes.c_void_p))
    ]

DaphneLib = ctypes.CDLL(os.path.join(PROTOTYPE_PATH, DAPHNELIB_FILENAME))
DaphneLib.getResult.restype = DaphneLibResult
