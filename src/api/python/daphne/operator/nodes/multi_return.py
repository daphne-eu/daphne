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

from daphne.operator.operation_node import OperationNode
from daphne.operator.nodes.frame import Frame
from daphne.operator.nodes.matrix import Matrix
from daphne.operator.nodes.scalar import Scalar
from daphne.script_building.dag import OutputType
from daphne.utils.consts import VALID_INPUT_TYPES, VALID_COMPUTED_TYPES
from daphne.utils.helpers import create_params_string
from daphne.utils.analyzer import get_argument_types
from daphne.script_building.nested_script import NestedDaphneDSLScript

import textwrap
from typing import TYPE_CHECKING, Union, Dict, Iterable, Sequence, Callable, Tuple, get_type_hints

if TYPE_CHECKING:
    # to avoid cyclic dependencies during runtime
    from daphne.context.daphne_context import DaphneContext

class MultiReturn(OperationNode):
    _outputs: Iterable['OperationNode']

    def __init__(self, daphne_context: 'DaphneContext', operation:str, output_nodes: Iterable[VALID_INPUT_TYPES], unnamed_input_nodes: Union[str, Iterable[VALID_INPUT_TYPES]]=None, 
                named_input_nodes: Dict[str, VALID_INPUT_TYPES]=None):
        """
        Operational node that represents calling a function with potentially multiple return values.
        Its reserved variable is left unused and is not added ot the generated script, however
        this variable name is used for a prefix for the return value nodes.

        :param daphne_context:
        :output_nodes: nodes reserved for representing the return values
        :param unnamed_input_nodes: unnamed nodes to be used as arguments
        :param named_input_nodes: named nodes to be used as arguments
        """
        self._outputs = output_nodes
        # set this node to be a source node for each retrun value node
        for node in self._outputs:
            node._source_node = self
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
        output_list=list()
        for idx, output_node in enumerate(self._outputs):
            name = f'{var_name}_{idx}'
            output_node.daphnedsl_name = name
            output_list.append(name)
        output_str = ",".join(output_list)
        return f'{output_str}={self.operation}({inputs_comma_sep});'
    
    def compute(self) -> None:
        raise NotImplementedError("'MultiReturn' node is not intended to be computed")

    @staticmethod
    def define_function(context: 'DaphneContext', callback: Callable[..., Tuple[VALID_COMPUTED_TYPES]]) -> Tuple[str, Iterable[VALID_COMPUTED_TYPES]]:
        """
        Generate DaphneDSL function defintion.

        :param context : 'DaphneContext' for storing the function definition
        :param callback : function for lazy evaluation
        :return : a tuple containing:
            - generated function name
            - tuple of operation nodes for the outputs
        """
        # intiate input nodes (arguments) for generating the function definition
        input_nodes = list()
        # dict with the same argument sequence as the callback signature
        arg_types = get_argument_types(callback)
        function_name = f"function_{len(context._functions.keys())}"
        for i, arg_type in enumerate(arg_types.values()):
            new_matrix = None
            if arg_type is None or arg_type == Matrix:
                new_matrix = Matrix(context, None)
            elif arg_type == Frame:
                new_matrix = Frame(context, None)
            elif arg_type == Scalar:
                new_matrix = Scalar(context, None)
            else:
                raise ValueError(f"Not allowed type hint {arg_type} for the 'dctx.function' signature")
            new_matrix.daphnedsl_name = f"ARG_{i}"
            input_nodes.append(new_matrix)
        # initiate return/output nodes to represent the return values for generating the function definition
        callback_outputs = callback(*input_nodes)        
        if (not isinstance(callback_outputs, tuple)):
            callback_outputs = (callback_outputs, )
        # check if all return values are valid types
        for node in callback_outputs:
            if not isinstance(node, (Matrix, Frame, Scalar)):
                raise ValueError(f"Not valid output node type {type(node)}")
        # generate the script witht the function definition
        script = NestedDaphneDSLScript(context, 1, 'F')
        names = script.build_code(callback_outputs)
        function_definition = str()
        function_definition += f"def {function_name}({','.join(map(lambda x: x.daphnedsl_name, input_nodes))}) {{\n"
        function_definition += textwrap.indent(script.daphnedsl_script, prefix="    ")
        function_definition += f"    return {','.join(names)};\n"
        function_definition += "}\n"
        # store the definition at the context
        context._functions[function_name] = function_definition
        return function_name, callback_outputs


