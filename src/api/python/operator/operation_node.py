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
import ctypes
from typing import Dict, Iterable, Optional, Sequence, Union, TYPE_CHECKING
from api.python.script_building.dag import DAGNode, OutputType
from api.python.script_building.script import DaphneDSLScript
from api.python.utils.consts import BINARY_OPERATIONS, F32, F64, SI32, SI64, SI8, UI32, UI64, UI8, VALID_INPUT_TYPES, DaphneLibResult, libDaphneShared
from api.python.utils.helpers import create_params_string
import numpy as np
import pandas as pd
if TYPE_CHECKING:
    # to avoid cyclic dependencies during runtime
    from context.daphne_context import DaphneContext
    
class OperationNode(DAGNode):  
    _result_var:Optional[Union[float,np.array]]
    _script:Optional[DaphneDSLScript]
    _output_types: Optional[Iterable[VALID_INPUT_TYPES]]
    _source_node: Optional["DAGNode"]
    _brackets: bool
    data: Optional[Union[pd.DataFrame,np.array]]

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

    def compute(self):
        if self._result_var is None:
            self._script = DaphneDSLScript(self.daphne_context)
            result = self._script.build_code(self)
            self._script.execute()
            #self._script.clear(self)
            if self._output_type == OutputType.FRAME:
                df = pd.read_csv(result)
                f = open(result+".meta")
                fmd = f.read().split(",")
                df.columns=fmd[1+2+int(fmd[1]):]
                result = df
            if self._output_type == OutputType.MATRIX:  
                #result = np.genfromtxt(result, delimiter=',')
               # libDaphneShared = ctypes.CDLL("build/src/api/shared/libDaphneShared.so")
                libDaphneShared.getResult.restype = DaphneLibResult
                daphneLibResult = libDaphneShared.getResult()
                result = np.ctypeslib.as_array(daphneLibResult.address)#, shape=[daphneLibResult.rows,daphneLibResult.cols]) 
                
                result = result.astype("float64")
            if result is None:
                return
            return result


    def code_line(self, var_name: str, unnamed_input_vars: Sequence[str], named_input_vars: Dict[str, str])->str:
        if self._brackets:
            return f'{var_name}={unnamed_input_vars[0]}[{",".join(unnamed_input_vars[1:])}]'
        if self.operation in BINARY_OPERATIONS:
            assert len(
                named_input_vars) == 0, 'named parameters can not be used with binary operations'
            assert len(unnamed_input_vars)==2, 'Binary operations need exactly two input variables'
            return f'{var_name}={unnamed_input_vars[0]}{self.operation}{unnamed_input_vars[1]};'
        inputs_comma_sep = create_params_string(unnamed_input_vars, named_input_vars).format(file_name=var_name)
        if self.output_type == OutputType.NONE:
            return f'{self.operation}({inputs_comma_sep});'
        else:
            return f'{var_name}={self.operation}({inputs_comma_sep});'


    def getType(self, vtc):
        if vtc == F64:
            return "float64"
        elif vtc == F32:
            return "float32"
        elif vtc == SI32:
            return "int32"
        elif vtc == SI8:
            return "int8"
        elif vtc == SI64:
            return "int64"
        elif vtc == UI32:
            return "uint32"
        elif vtc == UI8:
            return "uint8"
        elif vtc == UI64:
            return "uint64"
