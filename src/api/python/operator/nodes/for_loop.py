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
from api.python.operator.nodes.scalar import Scalar
from api.python.script_building.dag import OutputType
from api.python.utils.consts import VALID_INPUT_TYPES
from api.python.script_building.nested_script import NestedDaphneDSLScript
from api.python.utils import analyzer

from typing import TYPE_CHECKING, Dict, Iterable, Sequence, Tuple, Callable
import textwrap
from copy import copy

if TYPE_CHECKING:
    # to avoid cyclic dependencies during runtime
    from context.daphne_context import DaphneContext

class ForLoop(OperationNode):
    def __init__(self, daphne_context: 'DaphneContext', callback: Callable,
                 unnamed_input_nodes: Iterable[VALID_INPUT_TYPES] = None,
                 named_input_nodes: Iterable[VALID_INPUT_TYPES] = None) -> 'ForLoop':
        self.nested_level = 0  # default value
        _named_input_nodes = copy(named_input_nodes)
        _unnamed_input_nodes = copy(unnamed_input_nodes)

        if analyzer.get_number_argument(callback) != (len(unnamed_input_nodes) + 1):
            raise ValueError(f"{callback} does not have the same number of arguments as input nodes + 1")

        # define the variable for iteration with value 0 since declaration is not supported by DaphneDSL
        self.iter_num = Scalar(self, "0", assign=True)
        _named_input_nodes.update({"iter": self.iter_num})

        self.callback = lambda: callback(*unnamed_input_nodes, _named_input_nodes['iter'])

        outer_vars = analyzer.get_outer_scope_variables(callback)
        for node in outer_vars.values():
            if node:
                _unnamed_input_nodes.append(node)

        self._output = list()
        for node in unnamed_input_nodes:
            new_matrix_node = Matrix(self, None, [node], copy=True)
            new_matrix_node._source_node = self
            self._output.append(new_matrix_node)
        
        super().__init__(daphne_context, 'for_loop', unnamed_input_nodes=_unnamed_input_nodes, named_input_nodes=_named_input_nodes,
                         output_type=OutputType.NONE)

    def get_output(self) -> Tuple['Matrix']:
        return tuple(self._output)
        

    def code_line(self, var_name: str, unnamed_input_vars: Sequence[str],
                  named_input_vars: Dict[str, str]) -> str:
        callback_outputs = self.callback()
        if (not isinstance(callback_outputs, tuple)):
            callback_outputs = (callback_outputs, )
        script = NestedDaphneDSLScript(self.daphne_context, self.nested_level + 1)
        names = script.build_code(callback_outputs)

        step = str()
        if named_input_vars['step'] != "None":
            step = f":{named_input_vars['step']}"
        
        body = script.daphnedsl_script
        for i, name in enumerate(names):
            body += f"{unnamed_input_vars[i]}={name};\n"

        multiline_str = str()
        multiline_str += f"for({named_input_vars['iter']} in {named_input_vars['start']}:{named_input_vars['end']}{step}) {{\n"
        multiline_str += textwrap.indent(body, prefix="    ")
        multiline_str += "}"

        return multiline_str

    def compute(self) -> None:
        raise NotImplementedError("'ForLoop' node is not intended to be computed")