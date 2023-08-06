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

from api.python.operator.operation_node import OperationNode
from api.python.operator.nodes.frame import Frame
from api.python.operator.nodes.matrix import Matrix
from api.python.operator.nodes.scalar import Scalar
from api.python.script_building.dag import OutputType
from api.python.utils.consts import VALID_INPUT_TYPES, VALID_COMPUTED_TYPES
from api.python.script_building.script import DaphneDSLScript
from api.python.operator.nodes.scalar import Scalar
from api.python.script_building.dag import OutputType
from api.python.utils.consts import VALID_INPUT_TYPES, VALID_ARITHMETIC_TYPES, BINARY_OPERATIONS
from api.python.utils.daphnelib import DaphneLib, DaphneLibResult
from api.python.utils.helpers import create_params_string
from api.python.utils.analyzer import get_argument_types
from api.python.script_building.nested_script import NestedDaphneDSLScript

import textwrap
from typing import TYPE_CHECKING, Union, Dict, Iterable, Sequence, Callable, Tuple, get_type_hints

if TYPE_CHECKING:
    # to avoid cyclic dependencies during runtime
    from context.daphne_context import DaphneContext
import ctypes

if TYPE_CHECKING:
    # to avoid cyclic dependencies during runtime
    from context.daphne_context import DaphneContext

class MultiReturn(OperationNode):
    _outputs: Iterable['OperationNode']

    def __init__(self, daphne_context, operation:str,output_nodes, unnamed_input_nodes:Union[str, Iterable[VALID_INPUT_TYPES]]=None, 
                named_input_nodes:Dict[str, VALID_INPUT_TYPES]=None):
        self._outputs = output_nodes
        super().__init__(daphne_context, operation, unnamed_input_nodes, named_input_nodes, OutputType.MULTI_RETURN, False)
    
    def __getitem__(self, index):
        return self._outputs[index]

    def code_line(self, var_name: str, unnamed_input_vars: Sequence[str],
                  named_input_vars: Dict[str, str]) -> str:
        """
        Generates the code line for calling a function with potentially multiple return values.
        Here the names of the return/output nodes are derived by the 'var_name' of the multi-return node.

        :param var_name: variable name reserved for the operation node - NOT used
        :param unnamed_input_vars:
        :param named_input_vars:
        :return:
        """
        inputs_comma_sep = create_params_string(
            unnamed_input_vars, named_input_vars)
        output=""
        for idx, output_node in enumerate(self._outputs):
            name = f'{var_name}_{idx}'
            output_node.daphnedsl_name = name
            output += f'{name},'
        output = output[:-1]
        print(f'{output}={self.operation}({inputs_comma_sep})')
        return f'{output}={self.operation}({inputs_comma_sep});'

def compute(self, type="shared memory", num_returns = 2, verbose=False):
        """
        Compute function for processing the Daphne Object or operation node and returning the results.
        The function builds a DaphneDSL script from the node and its context, executes it, and processes the results
        to produce a pandas DataFrame, numpy array, or tensors.

        :param type: Execution type, either "shared memory" for in-memory data transfer or "files" for file-based data transfer.
        :param verbose: If True, outputs verbose logs, including timing information for each step.

        :return: Depending on the parameters and the operation's output type, this function can return:
            - A pandas DataFrame for frame outputs.
            - A numpy array for matrix outputs.
            - A scalar value for scalar outputs.
            - TensorFlow or PyTorch tensors if `isTensorflow` or `isPytorch` is set to True respectively.
        """
        if self._result_var is None:

            self._script = DaphneDSLScript(self.daphne_context)
            print(self)
            result = self._script.build_code(self, type, num_returns=num_returns)

            # Still a hard copy function that creates tmp files to execute
            self._script.executeSQL(multiText=True)
            self._script.clear(self)

            if result is None:
                return
            return result