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

from api.python.script_building.dag import DAGNode, OutputType
from api.python.script_building.script import DaphneDSLScript
from api.python.utils.consts import BINARY_OPERATIONS, TMP_PATH, VALID_INPUT_TYPES, F64, F32, SI64, SI32, SI8, UI64, UI32, UI8
from api.python.utils.daphnelib import DaphneLib, DaphneLibResult
from api.python.utils.helpers import create_params_string

import numpy as np
import pandas as pd
# import tensorflow as tf

import time

import ctypes
import json
import os
from typing import Dict, Iterable, Optional, Sequence, Union, TYPE_CHECKING

if TYPE_CHECKING:
    # to avoid cyclic dependencies during runtime
    from context.daphne_context import DaphneContext
    
class OperationNode(DAGNode):  
    _result_var:Optional[Union[float,np.array]]
    _script:Optional[DaphneDSLScript]
    _output_types: Optional[Iterable[VALID_INPUT_TYPES]]
    _source_node: Optional["DAGNode"]
    _brackets: bool
    data: Optional[Union[pd.DataFrame, np.array]]

    def __init__(self, daphne_context,operation:str, 
                unnamed_input_nodes: Union[str, Iterable[VALID_INPUT_TYPES]]=None,
                named_input_nodes: Dict[str, VALID_INPUT_TYPES]=None, 
                output_type:OutputType = OutputType.MATRIX, is_python_local_data: bool = False,
                brackets: bool = False):
        if unnamed_input_nodes is None:
            unnamed_input_nodes = []
        if named_input_nodes is None:
            named_input_nodes = []
        self.daphne_context = daphne_context
        self.operation = operation
        self._unnamed_input_nodes = unnamed_input_nodes
        self._named_input_nodes = named_input_nodes
        self._result_var = None
        self._script = None
        self._source_node = None
        self._already_added = False
        self.daphnedsl_name = ""
        self._is_python_local_data = is_python_local_data
        self._brackets = brackets
        self._output_type = output_type

    def compute(self, type="shared memory", isTensor=False, verbose=False):
        if self._result_var is None:
            
            if(verbose):
                # Time the execution for the whole processing
                start_time = time.time()

            self._script = DaphneDSLScript(self.daphne_context)
            for definition in self.daphne_context._functions.values():
                self._script.daphnedsl_script += definition
            result = self._script.build_code(self, type)
            
            if(verbose):
                # Time the execution for the execute function
                exec_start_time = time.time()

            # Still a hard copy function that creates tmp files to execute
            self._script.execute()
            self._script.clear(self)
<<<<<<< HEAD

            if(verbose):
                # Print the overall timing
                exec_end_time = time.time()
                print(f"Execute function Execution time: \n{exec_end_time - exec_start_time} seconds\n")

            if self._output_type == OutputType.FRAME and type=="shared memory":

                if(verbose):
                    # Time the execution for the compute function
                    comp_start_time = time.time()

                daphneLibResult = DaphneLib.getResult()

                data = []
                
                # Read the frame's address into a numpy array
                if daphneLibResult.columns is not None:

                    # Read the column labels and dtypes from the Frame's labels and dtypes directly
                    labels = [ctypes.cast(daphneLibResult.labels[i], ctypes.c_char_p).value.decode() for i in range(daphneLibResult.cols)]
                    
                    VTArray = ctypes.c_int64 * daphneLibResult.cols  # create a new type representing an array of data type codes
                    vtcs_array = ctypes.cast(daphneLibResult.vtcs, ctypes.POINTER(VTArray)).contents  # cast the pointer to this type and access its contents
                    dtypes = [self.getNumpyType(vtc) for vtc in vtcs_array]  # Convert the Data Types into Numpy Data Types

                    for idx in range(daphneLibResult.cols): 
                        data.append(np.ctypeslib.as_array(
                            ctypes.cast(daphneLibResult.columns[idx], ctypes.POINTER(self.getType(daphneLibResult.vtcs[idx]))),
                            shape=[daphneLibResult.rows, 1]
                        ))
                else:
                    print("Error: NULL pointer access")
                    labels = []
                    dtypes = []

                # Convert the numpy array to a pandas DataFrame
                df = pd.DataFrame(np.concatenate(data, axis=1))
                # Set the column labels in the DataFrame
                df.columns = labels

                # Cast each column to its appropriate dtype
                for col, dtype in zip(labels, dtypes):
                    df[col] = df[col].astype(dtype)

                result = df
                self.clear_tmp()

                if(verbose):
                    # Print the compute function timing
                    comp_end_time = time.time()
                    print(f"Compute Function Execution time: \n{comp_end_time - comp_start_time} seconds\n")

            elif self._output_type == OutputType.FRAME and type=="files":
=======
            if self._output_type == OutputType.FRAME:
>>>>>>> parent of dbd6c738 (CleanUp)
                df = pd.read_csv(result)
                with open(result + ".meta", "r") as f:
                    fmd = json.load(f)
                    df.columns = [x["label"] for x in fmd["schema"]]
                result = df
                self.clear_tmp()
            elif self._output_type == OutputType.MATRIX and type=="shared memory":
                daphneLibResult = DaphneLib.getResult()
                result = np.ctypeslib.as_array(
                    ctypes.cast(daphneLibResult.address, ctypes.POINTER(self.getType(daphneLibResult.vtc))),
                    shape=[daphneLibResult.rows, daphneLibResult.cols]
                )
                self.clear_tmp()
            elif self._output_type == OutputType.MATRIX and type=="files":
                arr = np.genfromtxt(result, delimiter=',')
                self.clear_tmp()
                return arr
            elif self._output_type == OutputType.SCALAR:
                # We transfer scalars back to Python by wrapping them into a 1x1 matrix.
                daphneLibResult = DaphneLib.getResult()
                result = np.ctypeslib.as_array(
                    ctypes.cast(daphneLibResult.address, ctypes.POINTER(self.getType(daphneLibResult.vtc))),
                    shape=[daphneLibResult.rows, daphneLibResult.cols]
                )[0, 0]
                self.clear_tmp()
            """
            if isTensor:
                if(verbose):
                    # Time the execution for the whole processing
                    tensor_start_time = time.time()

                # If result is a DataFrame (Frame)
                if self._output_type == OutputType.FRAME:
                    # Extract the dimension indices from the column labels
                    dimension_indices = [list(map(int, col.split('_')[1:])) for col in df.columns]

                    # Convert to a numpy array for easy manipulation
                    dimension_indices = np.array(dimension_indices)

                    # Find the maximum value for each dimension, plus 1, to compute the original shape
                    original_shape = tuple(dimension_indices.max(axis=0) + 1)

                    # Add the first dimension (rows)
                    original_shape = (len(df),) + original_shape

                    # Reconstruct the tensor
                    reshaped_arr = df.to_numpy().reshape(original_shape)
                    result = tf.convert_to_tensor(reshaped_arr)

                # If result is a numpy array (Matrix)
                elif self._output_type == OutputType.MATRIX:
                    result = tf.convert_to_tensor(result)  # Convert to a tensor

                else: 
                    print("Couldn't convert this Daphne Type to a tensor...")
                    print("Creating exemplary Tensor as return object")
                    tensor = tf.constant(np.arange(27).reshape((3, 3, 3)))
                    result = tensor

                if(verbose):
                    # Print the tensor timing
                    tensor_end_time = time.time()
                    print(f"Tensor Transformation Execution time: \n{tensor_end_time - tensor_start_time} seconds\n")
            """
            if(verbose):
                # Print the overall timing
                end_time = time.time()
                print(f"Overall Execution time: \n{end_time - start_time} seconds\n")

            if result is None:
                return
            return result

    def clear_tmp(self):
       for f in os.listdir(TMP_PATH):
          os.remove(os.path.join(TMP_PATH, f))

    def code_line(self, var_name: str, unnamed_input_vars: Sequence[str], named_input_vars: Dict[str, str])->str:
        if self._brackets:
            return f'{var_name}={unnamed_input_vars[0]}[{",".join(unnamed_input_vars[1:])}];'
        if self.operation in BINARY_OPERATIONS:
            assert len(
                named_input_vars) == 0, 'named parameters can not be used with binary operations'
            assert len(unnamed_input_vars)==2, 'Binary operations need exactly two input variables'
            return f'{var_name}={unnamed_input_vars[0]} {self.operation} {unnamed_input_vars[1]};'
        inputs_comma_sep = create_params_string(unnamed_input_vars, named_input_vars).format(file_name=var_name)
        if self.output_type == OutputType.NONE:
            return f'{self.operation}({inputs_comma_sep});'
        else:
            return f'{var_name}={self.operation}({inputs_comma_sep});'

    def getType(self, vtc):
        if vtc == F64:
            return ctypes.c_double
        elif vtc == F32:
            return ctypes.c_float
        elif vtc == SI64:
            return ctypes.c_int64
        elif vtc == SI32:
            return ctypes.c_int32
        elif vtc == SI8:
            return ctypes.c_int8
        elif vtc == UI64:
            return ctypes.c_uint64
        elif vtc == UI32:
            return ctypes.c_uint32
        elif vtc == UI8:
            return ctypes.c_uint8
        else:
            raise RuntimeError(f"unknown value type code: {vtc}")
    
    #Convert Daphne Data Types into Numpy Data Types
    def getNumpyType(self, vtc):
        if vtc == F64:
            return np.float64
        elif vtc == F32:
            return np.float32
        elif vtc == SI64:
            return np.int64
        elif vtc == SI32:
            return np.int32
        elif vtc == SI8:
            return np.int8
        elif vtc == UI64:
            return np.uint64
        elif vtc == UI32:
            return np.uint32
        elif vtc == UI8:
            return np.uint8
        else:
            raise RuntimeError(f"unknown value type code: {vtc}")
