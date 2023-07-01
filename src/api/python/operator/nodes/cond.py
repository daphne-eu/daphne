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
from api.python.utils.consts import VALID_INPUT_TYPES
from api.python.script_building.nested_script import NestedDaphneDSLScript
from api.python.utils import analyzer

from typing import TYPE_CHECKING, Dict, Iterable, Sequence, Tuple, Callable
import textwrap
from copy import copy

if TYPE_CHECKING:
    # to avoid cyclic dependencies during runtime
    from context.daphne_context import DaphneContext

class Cond(OperationNode):
    def __init__(self, daphne_context: 'DaphneContext', pred: Callable, true_fn: Callable, false_fn: Callable, 
                 unnamed_input_nodes: Iterable[VALID_INPUT_TYPES] = None) -> 'Cond':
        _named_input_nodes = dict()
        _unnamed_input_nodes = copy(unnamed_input_nodes)

        if analyzer.get_number_argument(true_fn) != analyzer.get_number_argument(false_fn):
            raise ValueError(f"{true_fn} and {false_fn} do not have the same number of arguents")
        elif analyzer.get_number_argument(true_fn) != len(unnamed_input_nodes):
            raise ValueError(f"{true_fn} and {false_fn} do not have the same number of arguments as input nodes")
        elif analyzer.get_number_argument(pred) != 0:
            raise ValueError(f"{pred} has more then 0 arguments")
        elif isinstance(pred(), tuple):
            raise ValueError(f"{pred} has more then 1 return values")
        
        self._true_fn = true_fn(*unnamed_input_nodes)
        if (not isinstance(self._true_fn, tuple)):
            self._true_fn = (self._true_fn, )

        self._false_fn = false_fn(*unnamed_input_nodes)
        if (not isinstance(self._false_fn, tuple)):
            self._false_fn = (self._false_fn, )

        if len(self._true_fn) != len(self._false_fn):
            raise ValueError(f"{true_fn} and {false_fn} do not have the same number return values")
        elif len(self._true_fn) != len(unnamed_input_nodes):
            raise ValueError(f"{true_fn} and {false_fn} do not have the same number return values as input nodes")

        outer_vars_true = analyzer.get_outer_scope_variables(true_fn)
        outer_vars_false = analyzer.get_outer_scope_variables(false_fn)
        outer_vars_both = {**outer_vars_true, **outer_vars_false}
        for node in outer_vars_both.values():
            if node:
                _unnamed_input_nodes.append(node)
        
        outer_vars_pred = analyzer.get_outer_scope_variables(pred)
        for node in outer_vars_pred.values():
            if node:
                _unnamed_input_nodes.append(node)

        _named_input_nodes.update({'pred': pred()})

        self._output = list()
        for node in unnamed_input_nodes:
            new_matrix_node = Matrix(self, None, [node], copy=True)
            new_matrix_node._source_node = self
            self._output.append(new_matrix_node)
        
        super().__init__(daphne_context, 'cond', unnamed_input_nodes=_unnamed_input_nodes, named_input_nodes=_named_input_nodes,
                         output_type=OutputType.NONE)

    def get_output(self) -> Tuple['Matrix']:
        return tuple(self._output)
        
    def code_line(self, var_name: str, unnamed_input_vars: Sequence[str],
                  named_input_vars: Dict[str, str]) -> str:
        # var_name is reserved for the operation but never used

        parent_level = 1  # default
        if self._script:
            parent_level = self._script._nested_level

        true_script = NestedDaphneDSLScript(self.daphne_context, parent_level+1)
        true_names = true_script.build_code(self._true_fn)

        false_script = NestedDaphneDSLScript(self.daphne_context, parent_level+1)
        false_names = false_script.build_code(self._false_fn)

        true_body = true_script.daphnedsl_script
        for i, name in enumerate(true_names):
            true_body += f"{unnamed_input_vars[i]}={name};\n"
        
        false_body = false_script.daphnedsl_script
        for i, name in enumerate(false_names):
            false_body += f"{unnamed_input_vars[i]}={name};\n"

        multiline_str = str()
        multiline_str += f"if ({named_input_vars['pred']}) {{\n"
        multiline_str += textwrap.indent(true_body, prefix="    ")
        multiline_str += "}"
        multiline_str += " else {\n"
        multiline_str += textwrap.indent(false_body, prefix="    ")
        multiline_str += "}"

        return multiline_str 

    def compute(self) -> None:
        raise NotImplementedError("'Cond' node is not intended to be computed")