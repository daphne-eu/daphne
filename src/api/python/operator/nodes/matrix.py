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

__all__ = ["Matrix"]

from api.python.operator.operation_node import OperationNode
from api.python.operator.nodes.scalar import Scalar
from api.python.script_building.dag import OutputType
from api.python.utils.consts import VALID_INPUT_TYPES, VALID_ARITHMETIC_TYPES, BINARY_OPERATIONS, TMP_PATH

import numpy as np

import json
import os
from typing import Union, TYPE_CHECKING, Dict, Iterable, Optional, Sequence

if TYPE_CHECKING:
    # to avoid cyclic dependencies during runtime
    from context.daphne_context import DaphneContext
    
class Matrix(OperationNode):
    _np_array: np.array

    def __init__(self, daphne_context: 'DaphneContext', operation:str, unnamed_input_nodes:Union[str, Iterable[VALID_INPUT_TYPES]]=None, 
                named_input_nodes:Dict[str, VALID_INPUT_TYPES]=None, 
                local_data: np.array = None, brackets:bool = False)->'Matrix':
        is_python_local_data = False
        if local_data is not None:
           
            self._np_array = local_data
            is_python_local_data = True
        else:
            self._np_array = None
        super().__init__(daphne_context, operation, unnamed_input_nodes, named_input_nodes, OutputType.MATRIX,is_python_local_data, brackets)

    def code_line(self, var_name: str, unnamed_input_vars: Sequence[str],
                  named_input_vars: Dict[str, str]) -> str:
        code_line = super().code_line(var_name, unnamed_input_vars, named_input_vars).format(file_name=var_name, TMP_PATH = TMP_PATH)
        
        if self._is_numpy() and self.operation == "readMatrix":
            with open(TMP_PATH+"/"+var_name+".csv", "wb") as f:
                np.savetxt(f, self._np_array, delimiter=",")
            with open(TMP_PATH+"/"+var_name+".csv.meta", "w") as f:
                json.dump(
                    {
                        "numRows": np.shape(self._np_array)[0],
                        "numCols": np.shape(self._np_array)[1],
                        "valueType": self.getDType(self._np_array.dtype),
                    },
                    f, indent=2
                )
        return code_line

    def getDType(self, d_type):
        if d_type == np.dtype('f4'):
            return "f32"
        elif d_type == np.dtype('f8'):
            return "f64"
        elif d_type == np.dtype('si2'):
            return "si8"
        elif d_type == np.dtype('si4'):
            return "si32"
        elif d_type == np.dtype('si8'):
            return "si64"
        elif d_type == np.dtype('ui2'):
            return "ui8"
        elif d_type == np.dtype('ui4'):
            return "ui8"
        elif d_type == np.dtype('ui8'):
            return "ui8"
        else:
            print("Error")

    def _is_numpy(self) -> bool:
        return self._np_array is not None
    
    def compute(self, type="shared memory") -> Union[np.array]:
        if self._is_numpy():
            return self._np_array
        else:
            return super().compute(type)

    def __add__(self, other: VALID_ARITHMETIC_TYPES) -> 'Matrix':
        return Matrix(self.daphne_context, '+', [self, other])

    def __sub__(self, other: VALID_ARITHMETIC_TYPES) -> 'Matrix':
        return Matrix(self.daphne_context,'-', [self, other])

    def __mul__(self, other: VALID_ARITHMETIC_TYPES) -> 'Matrix':
        return Matrix(self.daphne_context, '*', [self, other])

    def __truediv__(self, other: VALID_ARITHMETIC_TYPES) -> 'Matrix':
        return Matrix(self.daphne_context, '/', [self, other])

    def __lt__(self, other) -> 'Matrix':
        return Matrix(self.daphne_context, '<', [self, other])

    def __rlt__(self, other) -> 'Matrix':
        return Matrix(self.daphne_context, '<', [other, self])

    def __le__(self, other) -> 'Matrix':
        return Matrix(self.daphne_context, '<=', [self, other])

    def __rle__(self, other) -> 'Matrix':
        return Matrix(self.daphne_context, '<=', [other, self])

    def __gt__(self, other) -> 'Matrix':
        return Matrix(self.daphne_context, '>', [self, other])

    def __rgt__(self, other) -> 'Matrix':
        return Matrix(self.daphne_context, '>', [other, self])

    def __ge__(self, other) -> 'Matrix':
        return Matrix(self.daphne_context, '>=', [self, other])

    def __rge__(self, other) -> 'Matrix':
        return Matrix(self.daphne_context, '>=', [other, self])

    def __eq__(self, other) -> 'Matrix':
        return Matrix(self.daphne_context, '==', [self, other])

    def __req__(self, other) -> 'Matrix':
        return Matrix(self.daphne_context, '==', [other, self])

    def __ne__(self, other) -> 'Matrix':
        return Matrix(self.daphne_context, '!=', [self, other])

    def __rne__(self, other) -> 'Matrix':
        return Matrix(self.daphne_context, '!=', [other, self])

    def __matmul__(self, other: 'Matrix') -> 'Matrix':
        return Matrix(self.daphne_context, '@', [self, other])

    def __getitem__(self,  pos):
        if not isinstance(pos, int):
            i, x = pos
            return Matrix(self.daphne_context,'',[self, i, x], brackets=True)

    def sum(self, axis: int = None) -> 'OperationNode':
        """Calculate sum of matrix.
        :param axis: can be 0 or 1 to do either row or column sums
        :return: `Matrix` representing operation
        """
        if axis is None:
            return Scalar(self.daphne_context,'sum', [self])
        return Matrix(self.daphne_context,'sum', [self, axis])
        
    def sqrt(self) -> 'OperationNode':
        """Calculate sqrt of matrix.
        :return: `Matrix` representing operation
        """
        return Matrix(self.daphne_context,'sqrt', [self])
    
    def aggMin(self, axis: int = None) -> 'OperationNode':
        if axis is not None:
            return Matrix(self.daphne_context, 'aggMin', [self, axis])
        return Scalar(self.daphne_context, 'aggMin', [self])

    def aggMax(self, axis: int = None) -> 'OperationNode':
        if axis is not None:
            return Matrix(self.daphne_context, 'aggMax', [self, axis])
        return Scalar(self.daphne_context, 'aggMax', [self])
    
    def mean(self, axis: int = None) -> 'OperationNode':
        if axis is not None:
            return Matrix(self.daphne_context, 'mean', [self, axis])
        return Scalar(self.daphne_context, 'mean', [self])

    def stddev(self, axis: int = None) -> 'OperationNode':
        if axis is not None:
            return Matrix(self.daphne_context, 'stddev', [self, axis])
        return Scalar(self.daphne_context, 'stddev', [self])
    
    def ncol(self) -> 'OperationNode':
        return Scalar(self.daphne_context, 'ncol', [self])

    def nrow(self) -> 'OperationNode':
        return Scalar(self.daphne_context, 'nrow', [self])

    def diagMatrix(self) -> 'OperationNode':
        return Matrix(self.daphne_context, 'diagMatrix', [self])
    
    def solve(self, other: 'Matrix') -> 'Matrix':
        return Matrix(self.daphne_context, 'solve', [self, other])

    def t(self) -> 'OperationNode':
        return Matrix(self.daphne_context, 't', [self])
        
    def max(self, other: 'Matrix') -> 'Matrix':
        """Calculate elementwise max of two matrices.
        :param other: another matrix
        :return: `Matrix` representing operation
        """
        return Matrix(self.daphne_context, 'max', [self, other])

    def min(self, other: 'Matrix') -> 'Matrix':
        """Calculate elementwise min of two matrices.
        :param other: another matrix
        :return: `Matrix` representing operation
        """
        return Matrix(self.daphne_context, 'min', [self, other])
        
    def write(self, file: str) -> 'OperationNode':
        return OperationNode(self.daphne_context, 'writeMatrix', [self,'\"'+file+'\"'], output_type=OutputType.NONE)
    
    def print(self):
        return OperationNode(self.daphne_context,'print',[self], output_type=OutputType.NONE)