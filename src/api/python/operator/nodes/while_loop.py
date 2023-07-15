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
from api.python.operator.nodes.matrix import Matrix
from api.python.script_building.dag import OutputType
from api.python.utils.consts import VALID_INPUT_TYPES, VALID_ARITHMETIC_TYPES
from api.python.script_building.nested_script import NestedDaphneDSLScript
from api.python.utils import analyzer

from typing import TYPE_CHECKING, Dict, Iterable, Sequence, Tuple, Callable
import textwrap
from copy import copy

if TYPE_CHECKING:
    # to avoid cyclic dependencies during runtime
    from context.daphne_context import DaphneContext

class WhileLoop(OperationNode):
    def __init__(self, daphne_context: 'DaphneContext', cond: Callable, callback: Callable,
                 unnamed_input_nodes: Iterable[VALID_INPUT_TYPES] = None) -> 'WhileLoop':
        self.nested_level = 0  # default value
        _named_input_nodes = dict()
        _unnamed_input_nodes = copy(unnamed_input_nodes)

        if analyzer.get_number_argument(cond) != analyzer.get_number_argument(callback):
            raise ValueError(f"{cond} and {callback} do not have the same number of arguments")
        elif analyzer.get_number_argument(callback) != len(unnamed_input_nodes):
            raise ValueError(f"{callback} does not have the same number of arguments as input nodes")

        self.callback = lambda: callback(*unnamed_input_nodes)
        self.cond = lambda: cond(*unnamed_input_nodes)

        outer_vars_cond = analyzer.get_outer_scope_variables(cond)
        for node in outer_vars_cond.values():
            if node:
                _unnamed_input_nodes.update(node)

        outer_vars = analyzer.get_outer_scope_variables(callback)
        for node in outer_vars.values():
            if node:
                _unnamed_input_nodes.append(node)

        self._output = list()
        for node in unnamed_input_nodes:
            new_matrix_node = Matrix(self, None, [node], copy=True)
            new_matrix_node._source_node = self
            self._output.append(new_matrix_node)

        super().__init__(daphne_context, 'while_loop', unnamed_input_nodes=_unnamed_input_nodes,
            named_input_nodes=_named_input_nodes, output_type=OutputType.NONE)

    def get_output(self) -> Tuple['Matrix']:
        return tuple(self._output)
        

    def code_line(self, var_name: str, unnamed_input_vars: Sequence[str],
                  named_input_vars: Dict[str, str]) -> str:
        callback_outputs = self.callback()
        if (not isinstance(callback_outputs, tuple)):
            callback_outputs = (callback_outputs, )
        callback_script = NestedDaphneDSLScript(self.daphne_context, self.nested_level)
        callback_names = callback_script.build_code(callback_outputs)
        
        cond_outputs = self.cond()
        if (not isinstance(cond_outputs, tuple)):
            cond_outputs = (cond_outputs, )
        cond_script = NestedDaphneDSLScript(self.daphne_context, self.nested_level + 1, 'C')
        cond_name = cond_script.build_code(cond_outputs)[0]

        callback_body = callback_script.daphnedsl_script
        for i, name in enumerate(callback_names):
            callback_body += f"{unnamed_input_vars[i]}={name};\n"

        cond_body = cond_script.daphnedsl_script

        multiline_str = str()
        multiline_str += cond_body
        multiline_str += f"while ({cond_name}) {{\n"
        multiline_str += textwrap.indent(callback_body, prefix="    ")
        multiline_str += textwrap.indent(cond_body, prefix="    ")
        multiline_str += "}"

        return multiline_str

    def compute(self) -> None:
        return super().compute()
