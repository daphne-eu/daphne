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

from api.python.operator.nodes.frame import Frame
from api.python.operator.nodes.matrix import Matrix
from api.python.operator.nodes.scalar import Scalar
from api.python.operator.nodes.for_loop import ForLoop
from api.python.operator.nodes.cond import Cond
from api.python.operator.nodes.while_loop import WhileLoop
from api.python.operator.nodes.do_while_loop import DoWhileLoop
from api.python.operator.nodes.multi_return import MultiReturn
from api.python.utils.consts import VALID_INPUT_TYPES, VALID_COMPUTED_TYPES, TMP_PATH, F64, F32, SI64, SI32, SI8, UI64, UI32, UI8

import numpy as np
import pandas as pd
import csv
import tensorflow as tf
import torch as torch 

import time
from itertools import repeat

from typing import Sequence, Dict, Union

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
        :return: The data from numpy as a Matrix.
        """
        if(verbose):
            # Time the execution for the whole processing
            start_time = time.time()

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
            
            if(verbose):
                # Print the overall timing
                end_time = time.time()
                print(f"Overall Execution time: \n{end_time - start_time} seconds\n")

            return Matrix(self, 'receiveFromNumpy', [upper, lower, mat.shape[0], mat.shape[1], vtc], local_data=mat)
        else:
            # Data transfer via a file.
            unnamed_params = ['"src/api/python/tmp/{file_name}.csv\"']
            named_params = []

            if(verbose):
                # Print the overall timing
                end_time = time.time()
                print(f"Overall Execution time: \n{end_time - start_time} seconds\n")

            return Matrix(self, 'readMatrix', unnamed_params, named_params, local_data=mat)

    def from_pandas(self, df: pd.DataFrame, shared_memory=True, verbose=False) -> Frame:
        """Generates a `DAGNode` representing a frame with data given by a pandas `DataFrame`.
        :param df: The pandas DataFrame.
        :param shared_memory: Whether to use shared memory data transfer (True) or not (False).
        :param verbose: Whether the execution time and further information should be output to the console.
        :return: A Frame
        """

        if(verbose):
            # Time the execution for the whole processing
            start_time = time.time()
            # Time the execution for the Pandas Frame Type Checks
            typeCheck_start_time = time.time()

        # Check for a Series and convert to DataFrame
        if isinstance(df, pd.Series):
            print("Handling of pandas Series is not implemented yet. Converting to a standard DataFrame.")
            df = df.to_frame()

        # Check for MultiIndex and convert to standard DataFrame
        elif isinstance(df, pd.MultiIndex):
            print("Handling of pandas MultiIndex DataFrame is not implemented yet. \nConverting to a standard DataFrame is not possible...")
            print("Proceeding with an empty DataFrame")
            df = pd.DataFrame({"a": [0], "b": [0.0]})

        # Check for sparse DataFrame and convert to standard DataFrame
        elif isinstance(df.dtypes, pd.SparseDtype) or any(isinstance(item, pd.SparseDtype) for item in df.dtypes):
            print("Handling of pandas Sparse DataFrame is not implemented yet. Converting to a standard DataFrame.")
            df = df.sparse.to_dense()

        # Check for Categorical data and convert to standard DataFrame
        elif df.select_dtypes(include=["category"]).shape[1] > 0:
            print("Handling of pandas Categorical DataFrame is not implemented yet. Converting to a standard DataFrame.")
            df = df.apply(lambda x: x.cat.codes if x.dtype.name == 'category' else x)

        if(verbose):
            # Print the Type Check timing
            typeCheck_end_time = time.time()
            print(f"Frame Type Check Execution time: \n{(typeCheck_end_time - typeCheck_start_time):.10f} seconds\n")
           
        if shared_memory:
         
           # Convert dataframe and labels to column arrays and label arrays
            mats = []

            if(verbose):
                # Time the execution for all columns
                frame_start_time = time.time()

            for idx, column in enumerate(df):
                
                if(verbose):
                    # Time the execution for each column
                    col_start_time = time.time()

                mat = df[column].values

                if mat.dtype == np.int16:
                    mat = mat.astype(np.int32)

                if(verbose):
                    #Check if this step was zero copy
                    print(f'\nOriginal df column "{column}" ({idx}) shares memory with new numpy array: \n{np.shares_memory(mat, df[column].values)}\n')
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
                    print(f'Unsupported numpy dtype in column "{column}" ({idx})')
                
                mats.append(Matrix(self, 'receiveFromNumpy', [upper, lower, len(mat), 1 , vtc], local_data=mat))

                if(verbose):
                    # Print out timing
                    col_end_time = time.time()
                    print(f'Execution time for column "{column}" ({idx}): \n{(col_end_time - col_start_time):.10f} seconds\n')
            
            if(verbose):
                # Print out frame timing
                frame_end_time = time.time()
                print(f"Execution time for all columns: \n{(frame_end_time - frame_start_time):.10f} seconds\n")

            labels = df.columns
            for label in labels: 
                labelstr = f'"{label}"'
                mats.append(labelstr)
            
            if(verbose):
                # Print the overall timing
                end_time = time.time()
                print(f"Overall Execution time: \n{(end_time - start_time):.10f} seconds\n")

            # Return the Frame
            return Frame(self, 'createFrame', unnamed_input_nodes=mats, local_data = df)
        
        else:
            # Data transfer via files.
            unnamed_params = ['"src/api/python/tmp/{file_name}.csv\"']
            named_params = []

            if(verbose):
                # Print the overall timing
                end_time = time.time()
                print(f"Overall Execution time: \n{(end_time - start_time)::.10f} seconds\n")

            return Frame(self, 'readFrame', unnamed_params, named_params, local_data=df, column_names=df.columns)    
    
    def from_tensorflow(self, tensor: tf.Tensor, shared_memory=True, verbose=False):
        if verbose:
            start_time = time.time()

        if len(tensor.shape) == 2:
            # 2D tensor, handle as a matrix
            mat = tensor.numpy()  # Convert to numpy array

            if verbose:
                end_time = time.time()
                print(f"TensorFlow Tensor Reshape Execution time: \n{end_time - start_time} seconds\n")

            return self.from_numpy(mat, shared_memory, verbose)  # Using the existing from_numpy method
        
        else:
            # N-dimensional tensor, reshape to 2D and handle as a frame

            # Create a list of indices for all dimensions beyond the first
            indices = [range(dim) for dim in tensor.shape[1:]]

            # Create the column labels by using itertools.product to iterate over all combinations of indices
            columns = [f'dim_{"_".join(map(str, idx))}' for idx in product(*indices)]

            # Reshape the tensor into 2D, maintaining the first dimension and flattening all other dimensions
            reshaped_tensor = tf.reshape(tensor, (tensor.shape[0], -1))

            if verbose:
                df_start_time = time.time()

            # Convert to pandas DataFrame
            np_array = reshaped_tensor.numpy()
            df = pd.DataFrame(np_array, columns=columns)

            if verbose:
                df_end_time = time.time()
                print(f"TensorFlow Tensor to Dataframe Execution time: \n{df_end_time - df_start_time} seconds\n")

            if verbose:
                end_time = time.time()
                print(f"TensorFlow Tensor Reshape Execution time: \n{end_time - start_time} seconds\n")

            return self.from_pandas(df, shared_memory, verbose)  # Using the existing from_pandas method
    
    def from_pytorch(self, tensor: torch.Tensor, shared_memory=True, verbose=False):
        if verbose:
            start_time = time.time()

        if tensor.dim() == 2:
            # 2D tensor, handle as a matrix
            mat = tensor.numpy()  # Convert to numpy array

            if verbose:
                end_time = time.time()
                print(f"PyTorch Tensor Reshape Execution time: \n{end_time - start_time} seconds\n")

            return self.from_numpy(mat, shared_memory, verbose)  # Using the existing from_numpy method
        
        else:
            # N-dimensional tensor, reshape to 2D and handle as a frame

            # Create a list of indices for all dimensions beyond the first
            indices = [range(dim) for dim in tensor.size()[1:]]

            # Create the column labels by using itertools.product to iterate over all combinations of indices
            columns = [f'dim_{"_".join(map(str, idx))}' for idx in product(*indices)]

            # Reshape the tensor into 2D, maintaining the first dimension and flattening all other dimensions
            reshaped_tensor = torch.reshape(tensor, (tensor.size(0), -1))

            if verbose:
                df_start_time = time.time()

            # Convert to pandas DataFrame
            np_array = reshaped_tensor.numpy()
            df = pd.DataFrame(np_array, columns=columns)

            if verbose:
                df_end_time = time.time()
                print(f"PyTorch Tensor to Dataframe Execution time: \n{df_end_time - df_start_time} seconds\n")

            if verbose:
                end_time = time.time()
                print(f"PyTorch Tensor Reshape Execution time: \n{end_time - start_time} seconds\n")

            return self.from_pandas(df, shared_memory, verbose)  # Using the existing from_pandas method

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
