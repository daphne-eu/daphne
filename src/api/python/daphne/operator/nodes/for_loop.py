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
from api.python.operator.nodes.frame import Frame
from api.python.operator.nodes.scalar import Scalar
from api.python.script_building.dag import OutputType
from api.python.utils.consts import VALID_INPUT_TYPES, TMP_PATH
from api.python.script_building.nested_script import NestedDaphneDSLScript
from api.python.utils import analyzer

from typing import TYPE_CHECKING, Dict, Iterable, Sequence, Tuple, Callable
import textwrap

if TYPE_CHECKING:
    # to avoid cyclic dependencies during runtime
    from context.daphne_context import DaphneContext

class ForLoop(OperationNode):
    def __init__(self, daphne_context: 'DaphneContext', callback: Callable,
                 unnamed_input_nodes: Iterable[VALID_INPUT_TYPES] = None,
                 named_input_nodes: Dict[str,VALID_INPUT_TYPES] = None) -> 'ForLoop':
        """
        Operational node that represents for-loop functionality.
        Its reserved variable is left unused and is not added ot the generated script.
        It initiates a new interation variable of type scalar that is assumed to be
        represented by the last argument at the passed callback.

        :param daphne_context:
        :param callback:
        :param unnamed_input_nodes:
        :param named_input_nodes:
        """
        self.nested_level = 0  # default value
        # cast to dict to ensure it is a dict and to avoid additional copying
        _named_input_nodes = dict(named_input_nodes)
        # cast the iterable to list for consistency and to avoid addiotnal copying
        _unnamed_input_nodes = list(unnamed_input_nodes)
        # analyze if the passed functions fulfill the requirements
        if analyzer.get_number_argument(callback) != (len(unnamed_input_nodes) + 1):
            raise ValueError(f"{callback} does not have the same number of arguments as input nodes + 1")

        # define the variable for iteration with value 0 since declaration is not supported by DaphneDSL
        self.iter_num = Scalar(self, "0", assign=True)
        _named_input_nodes.update({"iter": self.iter_num})
        # spare storing the arguments additionally by redefining the callback function
        self.callback = lambda: callback(*unnamed_input_nodes, _named_input_nodes['iter'])
        # get the variables in outer scope to the callback function
        outer_vars = analyzer.get_outer_scope_variables(callback)
        # append the outer scope variables to input nodes so these
        # can be defined upfront by the depth-first-search pass
        for node in outer_vars.values():
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
        
        super().__init__(daphne_context, 'for_loop', unnamed_input_nodes=_unnamed_input_nodes, named_input_nodes=_named_input_nodes,
                         output_type=OutputType.NONE)

    def __getitem__(self, index) -> Tuple['Matrix']:
        return self._outputs[index]

    def code_line(self, var_name: str, unnamed_input_vars: Sequence[str],
                  named_input_vars: Dict[str, str]) -> str:
        """
        Generates the DaphneDSL code block for for-loop statement.
        Here the 'callback' is being evaluated and then the code
        lines are generated and added inside the loop code structure.

        :param var_name: variable name reserved for the operation node - NOT used
        :param unnamed_input_vars:
        :param named_input_vars:
        :return:
        """
        # handle loop body evaluation and code generation
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

        # pack all code lines in the while-loop structure
        multiline_str = str()
        multiline_str += f"for({named_input_vars['iter']} in {named_input_vars['start']}:{named_input_vars['end']}{step}) {{\n"
        multiline_str += textwrap.indent(body, prefix="    ")
        multiline_str += "}"

        return multiline_str

    def compute(self) -> None:
        raise NotImplementedError("'ForLoop' node is not intended to be computed")