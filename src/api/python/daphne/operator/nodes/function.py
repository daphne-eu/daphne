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

__all__ = ["Function"]

from api.python.operator.operation_node import OperationNode
from api.python.operator.nodes.matrix import Matrix
from api.python.script_building.dag import OutputType
from api.python.script_building.nested_script import NestedDaphneDSLScript
from api.python.utils import analyzer
from api.python.utils.consts import VALID_INPUT_TYPES, VALID_ARITHMETIC_TYPES, BINARY_OPERATIONS, TMP_PATH

import numpy as np
from copy import copy
import textwrap
import json
import os
from typing import Union, TYPE_CHECKING, Dict, Iterable, Optional, Sequence, Callable, Tuple

if TYPE_CHECKING:
    # to avoid cyclic dependencies during runtime
    from context.daphne_context import DaphneContext
    
class Function(OperationNode):
    __num_args: int

    def __init__(self, daphne_context: 'DaphneContext', callback: Callable,
                 unnamed_input_nodes: Iterable[VALID_INPUT_TYPES] = None)->'Function':
        self.nested_level = 0  # default value
        _named_input_nodes = dict()
        _unnamed_input_nodes = copy(unnamed_input_nodes)

        
        outer_vars_callback = analyzer.get_outer_scope_variables(callback)
        for node in outer_vars_callback.values():
            if node:
                _unnamed_input_nodes.append(node)
        self._output = list()
        print(unnamed_input_nodes)
        for node in unnamed_input_nodes:
            new_matrix_node = Matrix(self, None, [node], copy=True)
            new_matrix_node._source_node = self
            self._output.append(new_matrix_node)
        self.callback = lambda: callback(*self._output)

        super().__init__(daphne_context, 'function', unnamed_input_nodes=_unnamed_input_nodes,
            named_input_nodes=_named_input_nodes, output_type=OutputType.NONE)
    
    def get_output(self) -> Tuple['Matrix']:
        return tuple(self._output)
    
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
        callback_body += "return "
        callback_body += ",".join(callback_names)
        callback_body += ";\n"

        # pack all code lines in the while-loop structure
        multiline_str = str()
        multiline_str += f"def foo({', '.join(named_input_vars.keys())}) {{\n"
        multiline_str += textwrap.indent(callback_body, prefix="    ")
        multiline_str += "}\n"
        multiline_str += f"foo();"

        return multiline_str


    def compute(self, type="shared memory") -> 'OperationNode':
        raise NotImplementedError

    def call(self, *args) -> 'OperationNode':
        if len(args) != self.__num_args:
            raise RuntimeError(f"Number of given arguments does NOT correspond to the number of the expected arguments: {self.__num_args} args expected")

        return OperationNode(self.daphne_context, self.operation , args, output_type=OutputType.NONE)