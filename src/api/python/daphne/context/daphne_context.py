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

__all__ = ["DaphneContext", "Matrix", "Frame", "Scalar"]

from daphne.operator.nodes.frame import Frame
from daphne.operator.nodes.matrix import Matrix
from daphne.operator.nodes.scalar import Scalar
from daphne.operator.nodes.for_loop import ForLoop
from daphne.operator.nodes.cond import Cond
from daphne.operator.nodes.while_loop import WhileLoop
from daphne.operator.nodes.do_while_loop import DoWhileLoop
from daphne.operator.nodes.multi_return import MultiReturn
from daphne.operator.operation_node import OperationNode
from daphne.utils.consts import VALID_INPUT_TYPES, VALID_COMPUTED_TYPES, TMP_PATH, F64, F32, SI64, SI32, SI8, UI64, UI32, UI8

import numpy as np
import pandas as pd
try:
    import torch as torch
except ImportError as e:
    torch = e
try:
    import tensorflow as tf
except ImportError as e:
    tf = e

import time
from typing import Sequence, Dict, Union, List, Callable, Tuple, Optional, Iterable

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
    
    def from_numpy(self, mat: np.array, shared_memory=True, verbose=False) -> Matrix:
        """Generates a `DAGNode` representing a matrix with data given by a numpy `array`.
        :param mat: The numpy array.
        :param shared_memory: Whether to use shared memory data transfer (True) or not (False).
        :param verbose: Whether to print timing information (True) or not (False).
        :return: The data from numpy as a Matrix.
        """

        if verbose:
            start_time = time.time()
        
        # Handle the dimensionality of the matrix.
        if mat.ndim == 1:
            rows = mat.shape[0]
            cols = 1
        elif mat.ndim == 2:
            rows, cols = mat.shape
        else:
            raise ValueError("input numpy array should be 1d or 2d")

        if shared_memory:
            # Data transfer via shared memory.
            address = mat.ctypes.data_as(np.ctypeslib.ndpointer(dtype=mat.dtype, ndim=1, flags='C_CONTIGUOUS')).value
            upper = (address & 0xFFFFFFFF00000000) >> 32
            lower = (address & 0xFFFFFFFF)

            # Change the data type, if int16 or uint16 is handed over.
            # TODO This could change the input DataFrame.
            if mat.dtype == np.int16:
                mat = mat.astype(np.int32, copy=False)
            elif mat.dtype == np.uint16:
                mat = mat.astype(np.uint32, copy=False)

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
                # TODO Raise an error here?
                print("unsupported numpy dtype")

            res = Matrix(self, 'receiveFromNumpy', [upper, lower, rows, cols, vtc], local_data=mat)
        else:
            # Data transfer via a file.
            data_path_param = "\"" + TMP_PATH + "/{file_name}.csv\""
            unnamed_params = [data_path_param]
            named_params = []

            res = Matrix(self, 'readMatrix', unnamed_params, named_params, local_data=mat)

        if verbose:
            print(f"from_numpy(): total Python-side execution time: {(time.time() - start_time):.10f} seconds")

        return res

    def from_pandas(self, df: pd.DataFrame, shared_memory=True, verbose=False, keepIndex=False) -> Frame:
        """Generates a `DAGNode` representing a frame with data given by a pandas `DataFrame`.
        :param df: The pandas DataFrame.
        :param shared_memory: Whether to use shared memory data transfer (True) or not (False).
        :param verbose: Whether the execution time and further information should be output to the console.
        :param keepIndex: Whether the frame should keep its index from pandas within DAPHNE
        :return: A Frame
        """

        if verbose:
            start_time = time.time()
        
        if keepIndex:
            # Reset the index, moving it to a new column.
            # TODO We should not modify the input data frame here.
            df.reset_index(drop=False, inplace=True)

        # Check for various special kinds of pandas data objects
        # and handle them accordingly.
        if isinstance(df, pd.Series):
            # Convert Series to standard DataFrame.
            df = df.to_frame()
        elif isinstance(df, pd.MultiIndex):
            # MultiIndex cannot be converted to standard DataFrame.
            raise TypeError("handling of pandas MultiIndex DataFrame is not implemented yet")
        elif isinstance(df.dtypes, pd.SparseDtype) or any(isinstance(item, pd.SparseDtype) for item in df.dtypes):
            # Convert sparse DataFrame to standard DataFrame.
            df = df.sparse.to_dense()
        elif df.select_dtypes(include=["category"]).shape[1] > 0:
            # Convert categorical DataFrame to standard DataFrame.
            df = df.apply(lambda x: x.cat.codes if x.dtype.name == "category" else x)

        if verbose:
            print(f"from_pandas(): Python-side type-check execution time: {(time.time() - start_time):.10f} seconds")
           
        if shared_memory: # data transfer via shared memory
            # Convert DataFrame and labels to column arrays and label arrays.
            args = []

            if verbose:
                frame_start_time = time.time()

            for idx, column in enumerate(df):
                if verbose:
                    col_start_time = time.time()

                mat = df[column].values
                
                # Change the data type, if int16 or uint16 is handed over.
                # TODO This could change the input DataFrame.
                if mat.dtype == np.int16:
                    mat = mat.astype(np.int32, copy=False)
                elif mat.dtype == np.uint16:
                    mat = mat.astype(np.uint32, copy=False)

                if verbose:
                    # Check if this step was zero copy.
                    print(f"from_pandas(): original DataFrame column `{column}` (#{idx}) shares memory with new numpy array: {np.shares_memory(mat, df[column].values)}")

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
                    raise TypeError(f'Unsupported numpy dtype in column "{column}" ({idx})')
                
                args.append(Matrix(self, 'receiveFromNumpy', [upper, lower, len(mat), 1 , vtc], local_data=mat))

                if verbose:
                    print(f"from_pandas(): Python-side execution time for column `{column}` (#{idx}): {(time.time() - col_start_time):.10f} seconds")
            
            if verbose:
                print(f"from_pandas(): Python-side execution time for all columns: {(time.time() - frame_start_time):.10f} seconds")

            labels = df.columns
            for label in labels: 
                labelstr = f'"{label}"'
                args.append(labelstr)
            
            if verbose:
                print(f"from_pandas(): total Python-side execution time: {(time.time() - start_time):.10f} seconds")

            # Return the Frame.
            return Frame(self, 'createFrame', unnamed_input_nodes=args, local_data=df)
        
        else: # data transfer via files
            data_path_param = "\"" + TMP_PATH + "/{file_name}.csv\""
            unnamed_params = [data_path_param]
            named_params = []

            if verbose:
                print(f"from_pandas(): total Python-side execution time: {(time.time() - start_time)::.10f} seconds")

            # Return the Frame.
            return Frame(self, 'readFrame', unnamed_params, named_params, local_data=df, column_names=df.columns)    
    
    # This feature is only available if TensorFlow is available.
    if isinstance(tf, ImportError):
        def from_tensorflow(self, tensor           , shared_memory=True, verbose=False, return_shape=False):
            raise tf
    else:
        def from_tensorflow(self, tensor: tf.Tensor, shared_memory=True, verbose=False, return_shape=False):
            """Generates a `DAGNode` representing a matrix with data given by a TensorFlow `Tensor`.
            :param tensor: The TensorFlow Tensor.
            :param shared_memory: Whether to use shared memory data transfer (True) or not (False).
            :param verbose: Whether the execution time and further information should be output to the console.
            :param return_shape: Whether the original shape of the input tensor shall be returned.
            :return: A Matrix or a tuple of a Matrix and the original tensor shape (if `return_shape == True`).
            """

            # Store the original shape for later use.
            original_shape = tensor.shape
            
            if verbose:
                start_time = time.time()

            # Check if the tensor is 2d or higher dimensional.
            if len(original_shape) == 2:
                # If 2d, handle as a matrix, convert to numpy array.
                # This function is only zero copy, if the tensor is shared within the CPU.
                mat = tensor.numpy()
                # Using the existing from_numpy method for 2d arrays.
                matrix = self.from_numpy(mat, shared_memory, verbose)
            else:
                # If higher dimensional, reshape to 2d and handle as a matrix.
                # Store the original numpy representation.
                original_tensor = tensor.numpy()
                # Reshape to 2d using numpy's zero copy reshape.
                reshaped_tensor = original_tensor.reshape((original_shape[0], -1))

                if verbose:
                    # Check if the original and reshaped tensors share memory.
                    shares_memory = np.shares_memory(tensor, reshaped_tensor)
                    print(f"from_tensorflow(): original and reshaped tensors share memory: {shares_memory}")

                # Use the existing from_numpy method for the reshaped 2D array
                matrix = self.from_numpy(mat=reshaped_tensor, shared_memory=shared_memory, verbose=verbose)

            if verbose:
                print(f"from_tensorflow(): total Python-side execution time: {(time.time() - start_time):.10f} seconds")

            # Return the matrix, and the original shape if return_shape is set to True.
            return (matrix, original_shape) if return_shape else matrix

    # This feature is only available if PyTorch is available.
    if isinstance(torch, ImportError):
        def from_pytorch(self, tensor              , shared_memory=True, verbose=False, return_shape=False):
            raise torch
    else:
        def from_pytorch(self, tensor: torch.Tensor, shared_memory=True, verbose=False, return_shape=False):
            """Generates a `DAGNode` representing a matrix with data given by a PyTorch `Tensor`.
            :param tensor: The PyTorch Tensor.
            :param shared_memory: Whether to use shared memory data transfer (True) or not (False).
            :param verbose: Whether the execution time and further information should be output to the console.
            :param return_shape: Whether the original shape of the input tensor shall be returned.
            :return: A Matrix or a tuple of a Matrix and the original tensor shape (if `return_shape == True`).
            """

            # Store the original shape for later use.
            original_shape = tensor.size()
            
            if verbose:
                start_time = time.time()

            # Check if the tensor is 2d or higher dimensional.
            if tensor.dim() == 2:
                # If 2d, handle as a matrix, convert to numpy array.
                # If the Tensor is stored on the CPU, mat = tensor.numpy(force=True) can speed up the performance.
                mat = tensor.numpy()
                # Using the existing from_numpy method for 2d arrays.
                matrix = self.from_numpy(mat, shared_memory, verbose)
            else:
                # If higher dimensional, reshape to 2d and handle as a matrix.
                # Store the original numpy representation.
                original_tensor = tensor.numpy(force=True)
                # Reshape to 2d
                # TODO Does this change the input tensor?
                reshaped_tensor = original_tensor.reshape((original_shape[0], -1))

                if verbose:
                    # Check if the original and reshaped tensors share memory and print the result.
                    shares_memory = np.shares_memory(original_tensor, reshaped_tensor)
                    print(f"from_pytorch(): original and reshaped tensors share memory: {shares_memory}")

                # Use the existing from_numpy method for the reshaped 2d array.
                matrix = self.from_numpy(mat=reshaped_tensor, shared_memory=shared_memory, verbose=verbose)

            if verbose:
                print(f"from_pytorch(): total execution time: {(time.time() - start_time):.10f} seconds")

            # Return the matrix, and the original shape if return_shape is set to True.
            return (matrix, original_shape) if return_shape else matrix

    def fill(self, arg, rows:int, cols:int) -> Matrix:
        named_input_nodes = {'arg':arg, 'rows':rows, 'cols':cols}
        return Matrix(self, 'fill', [], named_input_nodes=named_input_nodes)
    
    def createFrame(self, columns: List[Matrix], labels:List[str] = None) -> 'Frame':
        if labels is None:
            labels = []
        if len(labels) != 0 and len(columns) != len(labels):
            raise RuntimeError(
                "createFrame: specifying labels is optional, but if labels are given, "
                "then their number must match that of the given columns"
            )
        
        # If a label is a Python string, then wrap it into quotation marks, such that
        # is becomes a string literal in DaphneDSL.
        labels = list(map(lambda l: f'"{l}"' if isinstance(l, str) else l, labels))

        return Frame(self, 'createFrame', [*columns, *labels])
    
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
    
    def sample(self, range, size, withReplacement: bool, seed = -1) -> 'Matrix':
        return Matrix(self, 'sample', [range, size, withReplacement, seed])

    def diagMatrix(self, arg: Matrix) -> 'Matrix':
        return Matrix(self, 'diagMatrix', [arg])

    def for_loop(self, input_nodes: Iterable[VALID_COMPUTED_TYPES], callback: Callable, start: int, end: int, step: Optional[int] = None) -> Tuple[VALID_COMPUTED_TYPES]:
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
        return tuple(ForLoop(self, callback, input_nodes, named_input_nodes))

    def cond(self, input_nodes: Iterable[VALID_COMPUTED_TYPES], pred: Callable, then_fn: Callable, else_fn: Callable = None) -> Tuple[VALID_COMPUTED_TYPES]:
        """
        Generates an if-then-else statement block for lazy evaluation.
        The generated block/operation cannot be directly computed
        but any of the outputs can.
        :param input_nodes: matrices for manipulation
        :param pred: the predicate (0 arguments, 1 return value)
        :param then_fn: callable to be performed if pred evaluates to true (n arguments, n return values, n=[1, ...])
        :param else_fn: callable to be performed if pred evaluates to false (n arguments, n return values)
        :return: manipulated matrices (length n)
        """
        return tuple(Cond(self, pred, then_fn, else_fn, input_nodes))
    
    def while_loop(self, input_nodes: Iterable[VALID_COMPUTED_TYPES], cond: Callable, callback: Callable) -> Tuple[VALID_COMPUTED_TYPES]:
        """
        Generates a while-loop block for lazy evaluation.
        The generated block/operation cannot be directly computed
        but any of the outputs can.
        :param input_nodes: matrices for manipulation
        :param cond: the condition (n arguments, 1 return value)
        :param callback: callable to be performed as long as cond evaluates to true (n arguments, n return values, n=[1, ...])
        :return: manipulated matrices (length n)
        """
        return tuple(WhileLoop(self, cond, callback, input_nodes))
    
    def do_while_loop(self, input_nodes: Iterable[VALID_COMPUTED_TYPES], cond: Callable, callback: Callable) -> Tuple[VALID_COMPUTED_TYPES]:
        """
        Generates a do-while-loop block for lazy evaluation.
        The generated block/operation cannot be directly computed
        but any of the outputs can.
        :param input_nodes: matrices for manipulation
        :param cond: the condition (n arguments, 1 return value)
        :param callback: callable to be performed as long as cond evaluates to true (n arguments, n return values, n=[1, ...])
        :return: manipulated matrices (length n)
        """
        return tuple(DoWhileLoop(self, cond, callback, input_nodes))

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
        Generates a user-defined function for lazy evaluation. 
        The generated function cannot be directly computed
        but any of the outputs can by using indexing.

        :param callback: callable with user-defined instructions
        :return: output nodes (matrices, scalars or frames)
        """
        # generate function definition
        function_name, callback_outputs = MultiReturn.define_function(self, callback)
        # generate function for calling
        def dctx_function(*args):
            output_nodes = list()
            for node in callback_outputs:
                if isinstance(node, Matrix):
                    output_nodes.append(Matrix(self, ''))
                elif isinstance(node, Frame):
                    output_nodes.append(Frame(self, ''))
                elif isinstance(node, Scalar):
                    output_nodes.append(Scalar(self, ''))
            return tuple(MultiReturn(self, function_name, output_nodes, args))
        
        return dctx_function
    
    def sql(self, query) -> Frame: 
        """
        Forwards and executes a sql query in Daphne
        :param query: The full SQL Query to be executed
        :return: A Frame based on the SQL Result
        """
        query_str = f'"{query}"'

        return Frame(self, 'sql', [query_str])
