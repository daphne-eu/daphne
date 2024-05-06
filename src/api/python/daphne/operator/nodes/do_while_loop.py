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
from daphne.operator.nodes.matrix import Matrix
from daphne.operator.nodes.scalar import Scalar
from daphne.operator.nodes.frame import Frame
from daphne.script_building.dag import OutputType
from daphne.utils.consts import VALID_INPUT_TYPES, VALID_ARITHMETIC_TYPES
from daphne.script_building.nested_script import NestedDaphneDSLScript
from daphne.utils import analyzer

from typing import TYPE_CHECKING, Dict, Iterable, Sequence, Tuple, Callable
import textwrap
from copy import copy

if TYPE_CHECKING:
    # to avoid cyclic dependencies during runtime
    from daphne.context.daphne_context import DaphneContext

class DoWhileLoop(OperationNode):
    def __init__(self, daphne_context: 'DaphneContext', cond: Callable, callback: Callable,
                 unnamed_input_nodes: Iterable[VALID_INPUT_TYPES] = None) -> 'DoWhileLoop':
        """
        Operational node that represents do-while-loop functionality.
        Its reserved variable is left unused and is not added ot the generated script.

        :param daphne_context:
        :param cond: function representing the condition logic
        :param callback: function for the logic of the loop body
        :param unnamed_input_nodes: operation nodes that are up for manipulation
        """
        self.nested_level = 0  # default value
        # cast the iterable to list for consistency and to avoid addiotnal copying
        _unnamed_input_nodes = list(unnamed_input_nodes)
        # analyze if the passed functions fulfill the requirements
        if analyzer.get_number_argument(cond) != analyzer.get_number_argument(callback):
            raise ValueError(f"{cond} and {callback} do not have the same number of arguments")
        elif analyzer.get_number_argument(callback) != len(unnamed_input_nodes):
            raise ValueError(f"{callback} does not have the same number of arguments as input nodes")

        # spare storing the arguments additionally by redefining the functions
        self.callback = lambda: callback(*unnamed_input_nodes)
        self.cond = lambda: cond(*unnamed_input_nodes)
        # get the variables in outer scope to the according functions
        outer_vars_cond = analyzer.get_outer_scope_variables(cond)
        outer_vars_callback = analyzer.get_outer_scope_variables(callback)
        # append the outer scope variables to input nodes so these
        # can be defined upfront by the depth-first-search pass
        for node in outer_vars_cond.values():
            if node:
                _unnamed_input_nodes.update(node)
        for node in outer_vars_callback.values():
            if node:
                _unnamed_input_nodes.append(node)

        # TODO: decide if here is the best place for this piece of code: maybe just after the first analysis
        # initiate the output operation nodes
        self._outputs = list()
        for node in unnamed_input_nodes:
            new_matrix_node = None
            if isinstance(node, Matrix):
                new_matrix_node = Matrix(daphne_context, None, [node], copy=True)
            elif isinstance(node, Frame):
                new_matrix_node =  Frame(daphne_context, None, [node], copy=True)
            elif isinstance(node, Scalar):  
                new_matrix_node = Scalar(daphne_context, None, [node], copy=True)
            else:
                raise ValueError(f"Unsupported input node type {type(node)}")
            new_matrix_node._source_node = self
            self._outputs.append(new_matrix_node)

        super().__init__(daphne_context, 'do_while_loop', unnamed_input_nodes=_unnamed_input_nodes,
                         output_type=OutputType.NONE)

    def __getitem__(self, index) -> Tuple['Matrix']:
        return self._outputs[index]  

    def code_line(self, var_name: str, unnamed_input_vars: Sequence[str],
                  named_input_vars: Dict[str, str]) -> str:
        """
        Generates the DaphneDSL code block for do-while-loop statement.
        Here the 'callback' and 'cond' are being
        evaluated and then the code lines for their according functionalities
        are generated and added inside the loop code structure.

        :param var_name: variable name reserved for the operation node - NOT used
        :param unnamed_input_vars:
        :param named_input_vars:
        :return:
        """
        # handle loop body evaluation and code generation
        callback_outputs = self.callback()
        if not isinstance(callback_outputs, tuple):
            callback_outputs = (callback_outputs, )
        # TODO: check why here is used sef.nested_level without +1
        callback_script = NestedDaphneDSLScript(self.daphne_context, self.nested_level)
        callback_names = callback_script.build_code(callback_outputs)
        callback_body = callback_script.daphnedsl_script
        for i, name in enumerate(callback_names):
            callback_body += f"{unnamed_input_vars[i]}={name};\n"

        # handle loop condition evaluation and code generation
        cond_outputs = self.cond()
        if not isinstance(cond_outputs, tuple):
            cond_outputs = (cond_outputs, )
        cond_script = NestedDaphneDSLScript(self.daphne_context, self.nested_level + 1, 'C')
        cond_name = cond_script.build_code(cond_outputs)[0]
        cond_body = cond_script.daphnedsl_script

        # pack all code lines in the while-loop structure
        multiline_str = str()
        multiline_str += cond_body
        multiline_str += "do {\n"
        multiline_str += textwrap.indent(callback_body, prefix="    ")
        multiline_str += textwrap.indent(cond_body, prefix="    ")
        multiline_str += f"}} while ({cond_name});"

        return multiline_str

    def compute(self) -> None:
        return super().compute()
