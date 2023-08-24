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

__all__ = ["DaphneContext"]

from api.python.operator.nodes.frame import Frame
from api.python.operator.nodes.matrix import Matrix
from api.python.operator.nodes.scalar import Scalar
from api.python.operator.nodes.for_loop import ForLoop
from api.python.operator.nodes.cond import Cond
from api.python.operator.nodes.while_loop import WhileLoop
from api.python.operator.nodes.do_while_loop import DoWhileLoop
from api.python.operator.nodes.multi_return import MultiReturn
from api.python.utils.consts import VALID_INPUT_TYPES, TMP_PATH, F64, F32, SI64, SI32, SI8, UI64, UI32, UI8
from api.python.utils.analyzer import get_number_argument
from api.python.operator.nodes.matrix import Matrix
from api.python.script_building.nested_script import NestedDaphneDSLScript

import textwrap
import numpy as np
import pandas as pd

from typing import Union, Callable, List, Tuple, Optional

class DaphneContext(object):
    _functions: dict
    
    def __init__(self):
        self._functions = dict()

    def readMatrix(self, file: str) -> Matrix:
        """Reads a matrix from a file.
        :param file: The path to the file containing the data.
        :return: The data in the file as a Matrix.
        """
        unnamed_params = ['\"'+file+'\"']
        return Matrix(self, 'readMatrix', unnamed_params)
        
    def readFrame(self, file: str) -> Frame:
        """Reads a frame from a file.
        :param file: The path to the file containing the data.
        :return: The data in the file as a Frame.
        """
        unnamed_params = ['\"'+file+'\"']
        return Frame(self, 'readFrame', unnamed_params)
    
    def from_numpy(self, mat: np.array, shared_memory=True) -> Matrix:
        """Generates a `DAGNode` representing a matrix with data given by a numpy `array`.
        :param mat: The numpy array.
        :param shared_memory: Whether to use shared memory data transfer (True) or not (False).
        :return: The data from numpy as a Matrix.
        """
        
        if shared_memory:
            # Data transfer via shared memory.
            address = mat.ctypes.data_as(np.ctypeslib.ndpointer(dtype=mat.dtype, ndim=1, flags='C_CONTIGUOUS')).value
            upper = (address & 0xFFFFFFFF00000000) >> 32
            lower = (address & 0xFFFFFFFF)
            d_type = mat.dtype
            if d_type == np.double or d_type == np.float64:
                vtc = F64
            elif d_type == np.float32:
                vtc = F32
            elif d_type == np.int8:
                vtc = SI8
            elif d_type == np.int32:
                vtc = SI32
            elif d_type == np.int64:
                vtc = SI64
            elif d_type == np.uint8:
                vtc = UI8
            elif d_type == np.uint32:
                vtc = UI32
            elif d_type == np.uint64:
                vtc = UI64
            else:
                print("unsupported numpy dtype")
            
            return Matrix(self, 'receiveFromNumpy', [upper, lower, mat.shape[0], mat.shape[1], vtc], local_data=mat)
        else:
            # Data transfer via a file.
            unnamed_params = ['"src/api/python/tmp/{file_name}.csv\"']
            named_params = []
            return Matrix(self, 'readMatrix', unnamed_params, named_params, local_data=mat)
        
    def from_pandas(self, df: pd.DataFrame) -> Frame:
        """Generates a `DAGNode` representing a frame with data given by a pandas `DataFrame`.
        :param df: The pandas DataFrame.
        :param args: unnamed parameters
        :param kwargs: named parameters
        :return: A Frame
        """

        # Data transfer via files.
        unnamed_params = ['"src/api/python/tmp/{file_name}.csv\"']
        named_params = []
        return Frame(self, 'readFrame', unnamed_params, named_params, local_data=df)
    
    def fill(self, arg, rows:int, cols:int) -> Matrix:
        named_input_nodes = {'arg':arg, 'rows':rows, 'cols':cols}
        return Matrix(self, 'fill', [], named_input_nodes=named_input_nodes)
    
    def seq(self, start, end, inc) -> Matrix:
        named_input_nodes = {'start':start, 'end':end, 'inc':inc}
        return Matrix(self, 'seq', [], named_input_nodes=named_input_nodes)

    def rand(self,
             rows: int, cols: int,
             min: Union[float, int] = None, max: Union[float, int] = None,
             sparsity: Union[float, int] = 0,
             seed: Union[float, int] = 0
    ) -> Matrix:
        """Generates a matrix filled with random values.
        :param rows: number of rows
        :param cols: number of columns
        :param min: min value
        :param max: max value
        :param sparsity: fraction of non-zero values
        :param seed: seed for pseudo random number generation
        :return: A Matrix
        """
        # TODO Why should we validate here, happens later in DAPHNE.
        if rows < 0:
            raise ValueError("In rand statement, can only assign rows a long (integer) value >= 0 "
                            "-- attempted to assign value: {r}".format(r=rows))
        if cols < 0:
            raise ValueError("In rand statement, can only assign cols a long (integer) value >= 0 "
                            "-- attempted to assign value: {c}".format(c=cols))
        named_input_nodes = {
            'rows': rows, 'cols': cols, 'min': min, 'max':max, 'sparsity':sparsity, 'seed':seed}

        return Matrix(self,'rand', [], named_input_nodes=named_input_nodes)

    def for_loop(self, input_nodes: List['Matrix'], callback: Callable, start: int, end: int, step: Optional[int] = None) -> Tuple['Matrix']:
        """
        Generates a for-loop block for lazy evaluation.
        The generated block/operation cannot be directly computed
        but any of the outputs can.
        :param input_nodes: matrices for manipulation
        :param callback: body functionality (n+1 arguments, n return values, n=[1, ...])
        :param start
        :param end
        :param step
        :return: manipulated matrices (length n)
        """
        named_input_nodes = {
            "start": start, 
            "end": end,
            "step": step
        }
        node = ForLoop(self, callback, input_nodes, named_input_nodes)
        return node.get_output()

    def cond(self, input_nodes: List['Matrix'], pred: Callable, true_fn: Callable, false_fn: Callable = None) -> Tuple['Matrix']:
        """
        Generates a if-else statement block for lazy evaluation.
        The generated block/operation cannot be directly computed
        but any of the outputs can.
        :param input_nodes: matrices for manipulation
        :param pred: the predicate (0 arguments, 1 return value)
        :param true_fn: callable to be performed if pred evaluates to true (n arguments, n return values, n=[1, ...])
        :param false_fn: callable to be performed if pred evaluates to false (n arguments, n return values)
        :return: manipulated matrices (length n)
        """
        node = Cond(self, pred, true_fn, false_fn, input_nodes)
        return node.get_output()
    
    def while_loop(self, input_nodes: List['Matrix'], cond: Callable, callback: Callable) -> Tuple['Matrix']:
        """
        Generates a while-loop block for lazy evaluation.
        The generated block/operation cannot be directly computed
        but any of the outputs can.
        :param input_nodes: matrices for manipulation
        :param cond: the condition (n arguments, 1 return value)
        :param callback: callable to be performed as long as cond evaluates to true (n arguments, n return values, n=[1, ...])
        :return: manipulated matrices (length n)
        """
        node = WhileLoop(self, cond, callback, input_nodes)
        return node.get_output()
    
    def do_while_loop(self, input_nodes: List['Matrix'], cond: Callable, callback: Callable) -> Tuple['Matrix']:
        """
        Generates a do-while-loop block for lazy evaluation.
        The generated block/operation cannot be directly computed
        but any of the outputs can.
        :param input_nodes: matrices for manipulation
        :param cond: the condition (n arguments, 1 return value)
        :param callback: callable to be performed as long as cond evaluates to true (n arguments, n return values, n=[1, ...])
        :return: manipulated matrices (length n)
        """
        node = DoWhileLoop(self, cond, callback, input_nodes)
        return node.get_output()

    def logical_and(self, left_operand: 'Scalar', right_operand: 'Scalar'):
        """
        Logical AND operation for lazy evaluation. 
        :param left_operand
        :param right_operand
        :return new Scalar
        """
        return Scalar(self, '&&', [left_operand, right_operand])
    
    def logical_or(self, left_operand: 'Scalar', right_operand: 'Scalar'):
        """
        Logical OR operation for lazy evaluation. 
        :param left_operand
        :param right_operand
        :return new Scalar
        """
        return Scalar(self, '||', [left_operand, right_operand])
    
    def function(self, callback: Callable):
        """
        Generated user-defined function for lazy evaluation. 
        The generated function cannot be directly computed
        but any of the outputs can by using indexing.

        :param callback: callable with user-defined instructions
        :return: output nodes (matrices, scalars or frames)
        """
        # intiate input nodes (arguments) for generating the function definition
        input_nodes = list()
        num_args = get_number_argument(callback)
        function_name = f"function_{len(self._functions.keys())}"
        for i in range(num_args):
            new_matrix = Matrix(self, None)
            new_matrix.daphnedsl_name = f"ARG_{i}"
            input_nodes.append(new_matrix)
        # initiate return/output nodes to represent the return values for generating the function definition
        callback_outputs = callback(*input_nodes)
        if (not isinstance(callback_outputs, tuple)):
            callback_outputs = (callback_outputs, )
        # generate the script witht the function definition
        script = NestedDaphneDSLScript(self, 1, 'F')
        names = script.build_code(callback_outputs)
        function_definition = str()
        function_definition += f"def {function_name}({','.join(map(lambda x: x.daphnedsl_name, input_nodes))}) {{\n"
        function_definition += textwrap.indent(script.daphnedsl_script, prefix="    ")
        function_definition += f"    return {','.join(names)};\n"
        function_definition += "}\n"
        # store the definition at the context
        self._functions[function_name] = function_definition
        # generate function for calling
        def dctx_function(*args):
            return MultiReturn(self, function_name, [Matrix(self, '') for _ in range(len(callback_outputs))], args)
        
        return dctx_function