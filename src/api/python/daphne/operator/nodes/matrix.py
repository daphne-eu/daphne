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
from typing import Union, TYPE_CHECKING, Dict, Iterable, Optional, Sequence, List

if TYPE_CHECKING:
    # to avoid cyclic dependencies during runtime
    from context.daphne_context import DaphneContext
    
class Matrix(OperationNode):
    _np_array: np.array
    __copy: bool

    def __init__(self, daphne_context: 'DaphneContext', operation:str, unnamed_input_nodes:Union[str, Iterable[VALID_INPUT_TYPES]]=None, 
                named_input_nodes:Dict[str, VALID_INPUT_TYPES]=None, 
                local_data: np.array = None, brackets:bool = False, copy: bool = False)->'Matrix':
        self.__copy = copy
        is_python_local_data = False
        if local_data is not None:
           
            self._np_array = local_data
            is_python_local_data = True
        else:
            self._np_array = None
        super().__init__(daphne_context, operation, unnamed_input_nodes, named_input_nodes, OutputType.MATRIX,is_python_local_data, brackets)

    def code_line(self, var_name: str, unnamed_input_vars: Sequence[str],
                  named_input_vars: Dict[str, str]) -> str:
        if self.__copy:
            return f'{var_name}={unnamed_input_vars[0]};'
        
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
        return Matrix(self.daphne_context, '-', [self, other])

    def __mul__(self, other: VALID_ARITHMETIC_TYPES) -> 'Matrix':
        return Matrix(self.daphne_context, '*', [self, other])

    def __truediv__(self, other: VALID_ARITHMETIC_TYPES) -> 'Matrix':
        return Matrix(self.daphne_context, '/', [self, other])

    def __pow__(self, other: VALID_ARITHMETIC_TYPES) -> 'Matrix':
        # "**" in Python, "^" in DaphneDSL.
        return Matrix(self.daphne_context, '^', [self, other])

    def __mod__(self, other: VALID_ARITHMETIC_TYPES) -> 'Matrix':
        return Matrix(self.daphne_context, '%', [self, other])

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
        return Matrix(self.daphne_context, '>= ', [other, self])

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
        
    def abs(self) -> 'OperationNode':
        return Matrix(self.daphne_context, 'abs', [self])
        
    def sign(self) -> 'OperationNode':
        return Matrix(self.daphne_context, 'sign', [self])
        
    def exp(self) -> 'OperationNode':
        return Matrix(self.daphne_context, 'exp', [self])
        
    def ln(self) -> 'OperationNode':
        return Matrix(self.daphne_context, 'ln', [self])
    
    def sqrt(self) -> 'OperationNode':
        """Calculate sqrt of matrix.
        :return: `Matrix` representing operation
        """
        return Matrix(self.daphne_context,'sqrt', [self])
    
    def round(self) -> 'OperationNode':
        return Matrix(self.daphne_context, 'round', [self])
        
    def floor(self) -> 'OperationNode':
        return Matrix(self.daphne_context, 'floor', [self])
        
    def ceil(self) -> 'OperationNode':
        return Matrix(self.daphne_context, 'ceil', [self])
        
    def sin(self) -> 'OperationNode':
        return Matrix(self.daphne_context, 'sin', [self])
        
    def cos(self) -> 'OperationNode':
        return Matrix(self.daphne_context, 'cos', [self])
        
    def tan(self) -> 'OperationNode':
        return Matrix(self.daphne_context, 'tan', [self])
        
    def sinh(self) -> 'OperationNode':
        return Matrix(self.daphne_context, 'sinh', [self])
        
    def cosh(self) -> 'OperationNode':
        return Matrix(self.daphne_context, 'cosh', [self])
        
    def tanh(self) -> 'OperationNode':
        return Matrix(self.daphne_context, 'tanh', [self])
        
    def asin(self) -> 'OperationNode':
        return Matrix(self.daphne_context, 'asin', [self])
        
    def acos(self) -> 'OperationNode':
        return Matrix(self.daphne_context, 'acos', [self])
        
    def atan(self) -> 'OperationNode':
        return Matrix(self.daphne_context, 'atan', [self])
    
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
    
    def var(self, axis: int = None) -> 'OperationNode':
        if axis is not None:
            return Matrix(self.daphne_context, 'var', [self, axis])
        return Scalar(self.daphne_context, 'var', [self])

    def stddev(self, axis: int = None) -> 'OperationNode':
        if axis is not None:
            return Matrix(self.daphne_context, 'stddev', [self, axis])
        return Scalar(self.daphne_context, 'stddev', [self])
    
    def idxMin(self, axis: int) -> 'OperationNode':
        if axis is None:
            raise RuntimeError("parameter axis must be provided for idxMin")
        return Scalar(self.daphne_context, 'idxMin', [self, axis])
    
    def idxMax(self, axis: int) -> 'OperationNode':
        if axis is None:
            raise RuntimeError("parameter axis must be provided for idxMax")
        return Scalar(self.daphne_context, 'idxMax', [self, axis])
    
    def cumSum(self) -> 'OperationNode':
        return Scalar(self.daphne_context, 'cumSum', [self])
    
    def cumProd(self) -> 'OperationNode':
        return Scalar(self.daphne_context, 'cumProd', [self])
    
    def cumMin(self) -> 'OperationNode':
        return Scalar(self.daphne_context, 'cumMin', [self])
    
    def cumMax(self) -> 'OperationNode':
        return Scalar(self.daphne_context, 'cumMax', [self])
    
    def ncol(self) -> 'OperationNode':
        return Scalar(self.daphne_context, 'ncol', [self])

    def nrow(self) -> 'OperationNode':
        return Scalar(self.daphne_context, 'nrow', [self])

    def ncell(self) -> 'OperationNode':
        return Scalar(self.daphne_context, 'ncell', [self])

    def diagVector(self) -> 'OperationNode':
        return Matrix(self.daphne_context, 'diagVector', [self])
    
    def solve(self, other: 'Matrix') -> 'Matrix':
        return Matrix(self.daphne_context, 'solve', [self, other])

    def t(self) -> 'OperationNode':
        return Matrix(self.daphne_context, 't', [self])

    def reshape(self, numRows: int, numCols: int) -> 'OperationNode':
        return Matrix(self.daphne_context, 'reshape', [self, numRows, numCols])

    def cbind(self, other) -> 'Matrix':
        return Matrix(self.daphne_context, "cbind", [self, other])

    def rbind(self, other) -> 'Matrix':
        return Matrix(self.daphne_context, "rbind", [self, other])

    def reverse(self) -> 'OperationNode':
        return Matrix(self.daphne_context, 'reverse', [self])

    def lowerTri(self, diag: bool, values: bool) -> 'OperationNode':
        return Matrix(self.daphne_context, 'lowerTri', [self, diag, values])

    def upperTri(self, diag: bool, values: bool) -> 'OperationNode':
        return Matrix(self.daphne_context, 'upperTri', [self, diag, values])

    def replace(self, pattern, replacement) -> 'OperationNode':
        return Matrix(self.daphne_context, 'replace', [self, pattern, replacement])
        
    def pow(self, other: 'Matrix') -> 'Matrix':
        return Matrix(self.daphne_context, 'pow', [self, other])
        
    def log(self, other: 'Matrix') -> 'Matrix':
        return Matrix(self.daphne_context, 'log', [self, other])
        
    def mod(self, other: 'Matrix') -> 'Matrix':
        return Matrix(self.daphne_context, 'mod', [self, other])
        
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
        
    def outerAdd(self, other: 'Matrix') -> 'Matrix':
        return Matrix(self.daphne_context, 'outerAdd', [self, other])
        
    def outerSub(self, other: 'Matrix') -> 'Matrix':
        return Matrix(self.daphne_context, 'outerSub', [self, other])
        
    def outerMul(self, other: 'Matrix') -> 'Matrix':
        return Matrix(self.daphne_context, 'outerMul', [self, other])
        
    def outerDiv(self, other: 'Matrix') -> 'Matrix':
        return Matrix(self.daphne_context, 'outerDiv', [self, other])
        
    def outerPow(self, other: 'Matrix') -> 'Matrix':
        return Matrix(self.daphne_context, 'outerPow', [self, other])
        
    def outerLog(self, other: 'Matrix') -> 'Matrix':
        return Matrix(self.daphne_context, 'outerLog', [self, other])
        
    def outerMod(self, other: 'Matrix') -> 'Matrix':
        return Matrix(self.daphne_context, 'outerMod', [self, other])
        
    def outerMin(self, other: 'Matrix') -> 'Matrix':
        return Matrix(self.daphne_context, 'outerMin', [self, other])
        
    def outerMax(self, other: 'Matrix') -> 'Matrix':
        return Matrix(self.daphne_context, 'outerMax', [self, other])
        
    def outerAnd(self, other: 'Matrix') -> 'Matrix':
        return Matrix(self.daphne_context, 'outerAnd', [self, other])
        
    def outerOr(self, other: 'Matrix') -> 'Matrix':
        return Matrix(self.daphne_context, 'outerOr', [self, other])
        
    def outerXor(self, other: 'Matrix') -> 'Matrix':
        return Matrix(self.daphne_context, 'outerXor', [self, other])
        
    def outerConcat(self, other: 'Matrix') -> 'Matrix':
        return Matrix(self.daphne_context, 'outerConcat', [self, other])
        
    def outerEq(self, other: 'Matrix') -> 'Matrix':
        return Matrix(self.daphne_context, 'outerEq', [self, other])
        
    def outerNeq(self, other: 'Matrix') -> 'Matrix':
        return Matrix(self.daphne_context, 'outerNeq', [self, other])
        
    def outerLt(self, other: 'Matrix') -> 'Matrix':
        return Matrix(self.daphne_context, 'outerLt', [self, other])
        
    def outerLe(self, other: 'Matrix') -> 'Matrix':
        return Matrix(self.daphne_context, 'outerLe', [self, other])
        
    def outerGt(self, other: 'Matrix') -> 'Matrix':
        return Matrix(self.daphne_context, 'outerGt', [self, other])
        
    def outerGe(self, other: 'Matrix') -> 'Matrix':
        return Matrix(self.daphne_context, 'outerGe', [self, other])
    
    def order(self, colIdxs: List[int], ascs: List[bool], returnIndexes: bool) -> 'Matrix':
        if len(colIdxs) != len(ascs):
            raise RuntimeError("order: the lists given for parameters colIdxs and ascs must have the same length")
        return Matrix(self.daphne_context, 'order', [self, *colIdxs, *ascs, returnIndexes])
        
    def write(self, file: str) -> 'OperationNode':
        return OperationNode(self.daphne_context, 'writeMatrix', [self,'\"'+file+'\"'], output_type=OutputType.NONE)
    
    def print(self):
        return OperationNode(self.daphne_context,'print',[self], output_type=OutputType.NONE)