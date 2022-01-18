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

import os
from typing import Union, TYPE_CHECKING, Dict, Iterable, Optional, Sequence
from api.python.script_building.dag import OutputType
from api.python.utils.consts import VALID_INPUT_TYPES, VALID_ARITHMETIC_TYPES, BINARY_OPERATIONS
from api.python.operator.operation_node import OperationNode
from api.python.operator.nodes.scalar import Scalar
import numpy as np


class Matrix(OperationNode):
    _np_array: np.array

    def __init__(self, operation:str, unnamed_input_nodes:Union[str, Iterable[VALID_INPUT_TYPES]]=None, 
                named_input_nodes:Dict[str, VALID_INPUT_TYPES]=None, 
                local_data: np.array = None, brackets:bool = False)->'Matrix':
        is_python_local_data = False
        if local_data is not None:
           
            self._np_array = local_data
            is_python_local_data = True
        else:
            self._np_array = None
        super().__init__(operation, unnamed_input_nodes, named_input_nodes, OutputType.MATRIX,is_python_local_data, brackets)
    

    def code_line(self, var_name: str, unnamed_input_vars: Sequence[str],
                  named_input_vars: Dict[str, str]) -> str:
        code_line = super().code_line(var_name, unnamed_input_vars, named_input_vars).format(file_name=var_name)

        if self._is_numpy():
            if("test/api/python" in os.getcwd()):
                path = "../../../"
            else:
                path = ""
            with open(path+"src/api/python/tmp/"+var_name+".csv", "wb") as f:
                np.savetxt(f, self._np_array, delimiter=",")
                f.close()
        return code_line

  
        
        
    def _is_numpy(self) -> bool:
        return self._np_array is not None
    
    def compute(self) -> Union[np.array]:
        if self._is_numpy():
            return self._np_array
        else:
            return super().compute()

    def __add__(self, other: VALID_ARITHMETIC_TYPES) -> 'Matrix':
        return Matrix('+', [self, other])

    def __sub__(self, other: VALID_ARITHMETIC_TYPES) -> 'Matrix':
        return Matrix('-', [self, other])


    def __mul__(self, other: VALID_ARITHMETIC_TYPES) -> 'Matrix':
        return Matrix( '*', [self, other])

    def __truediv__(self, other: VALID_ARITHMETIC_TYPES) -> 'Matrix':
        return Matrix( '/', [self, other])

    def sum(self, axis: int = None) -> 'OperationNode':
        """Calculate sum of matrix.
        :param axis: can be 0 or 1 to do either row or column sums
        :return: `Matrix` representing operation
        """
        if axis == 0:
            return Matrix('colSums', [self])
        elif axis == 1:
            return Matrix('rowSums', [self])
        elif axis is None:
            return Scalar('sum', [self])
        raise ValueError(
            f"Axis has to be either 0, 1 or None, for column, row or complete {self.operation}")
        
    def sqrt(self) -> 'OperationNode':
        """Calculate sqrt of matrix.
        :return: `Matrix` representing operation
        """
  
        return Matrix('sqrt', [self])
     
        
    def mean(self, axis: int = None) -> 'OperationNode':
        """Calculate mean of matrix.
        :param axis: can be 0 or 1 to do either row or column means
        :return: `Matrix` representing operation
        """
        if axis == 0:
            return Matrix('colMeans', [self])
        elif axis == 1:
            return Matrix('rowMeans', [self])
        elif axis is None:
            return Scalar('mean', [self])
        raise ValueError(
            f"Axis has to be either 0, 1 or None, for column, row or complete {self.operation}")
    
    def max(self, axis: int = None) -> 'OperationNode':
        """Calculate max of matrix.
        :param axis: can be 0 or 1 to do either row or column aggregation
        :return: `Matrix` representing operation
        """
        if axis == 0:
            return Matrix( 'colMaxs', [self])
        elif axis == 1:
            return Matrix('rowMaxs', [self])
        elif axis is None:
            return Scalar('max', [self])
        raise ValueError(
            f"Axis has to be either 0, 1 or None, for column, row or complete {self.operation}")

    def min(self, axis: int = None) -> 'OperationNode':
        """Calculate max of matrix.
        :param axis: can be 0 or 1 to do either row or column aggregation
        :return: `Matrix` representing operation
        """
        if axis == 0:
            return Matrix('colMins', [self])
        elif axis == 1:
            return Matrix('rowMins', [self])
        elif axis is None:
            return Scalar('min', [self])
        raise ValueError(
            f"Axis has to be either 0, 1 or None, for column, row or complete {self.operation}")

    def var(self, axis: int = None) -> 'OperationNode':
        """Calculate variance of matrix.
        :param axis: can be 0 or 1 to do either row or column vars
        :return: `Matrix` representing operation
        """
        if axis == 0:
            return Matrix('colVars', [self])
        elif axis == 1:
            return Matrix('rowVars', [self])
        elif axis is None:
            return Scalar( 'var', [self])
        raise ValueError(
            f"Axis has to be either 0, 1 or None, for column, row or complete {self.operation}")

    def abs(self) -> 'Matrix':
        """Calculate absolute.
        :return: `Matrix` representing operation
        """
        return Matrix( 'abs', [self])
    
    def sin(self) -> 'Matrix':
        """Calculate sin.
        :return: `Matrix` representing operation
        """
        return Matrix( 'sin', [self])

    def cos(self) -> 'Matrix':
        """Calculate cos.
        :return: `Matrix` representing operation
        """
        return Matrix('cos', [self])

    def tan(self) -> 'Matrix':
        """Calculate tan.
        :return: `Matrix` representing operation
        """
        return Matrix('tan', [self])

    def asin(self) -> 'Matrix':
        """Calculate arcsin.
        :return: `Matrix` representing operation
        """
        return Matrix( 'asin', [self])

    def acos(self) -> 'Matrix':
        """Calculate arccos.
        :return: `Matrix` representing operation
        """
        return Matrix('acos', [self])

    def atan(self) -> 'Matrix':
        """Calculate arctan.
        :return: `Matrix` representing operation
        """
        return Matrix( 'atan', [self])
        
    def print(self):
        return OperationNode('print',[self], output_type=OutputType.NONE)
    
    def to_csv(self):
        return OperationNode( 'writeMatrix', [self, '"src/api/python/tmp/{file_name}.csv"'], output_type=OutputType.NONE)