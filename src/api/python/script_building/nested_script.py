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
from api.python.operator.nodes.matrix import Matrix
from api.python.operator.operation_node import OperationNode
from api.python.utils.consts import VALID_INPUT_TYPES, TMP_PATH, PROTOTYPE_PATH
from api.python.utils.daphnelib import DaphneLib
from api.python.script_building.script import DaphneDSLScript

import ctypes
import os
from typing import List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    # to avoid cyclic dependencies during runtime
    from context.daphne_context import DaphneContext

class NestedDaphneDSLScript(DaphneDSLScript):
    input_var: str 
    iter_var: str
    def __init__(self, context, nested_level=1) -> None:
        super().__init__(context)
        self._nested_level = nested_level

    def build_code(self, dag_root: DAGNode):
        self._dfs_dag_nodes(dag_root)

    def execute(self):
        raise TypeError("NestedDaphneDSLScript can not be executed!")

    def _dfs_dag_nodes(self, dag_node: VALID_INPUT_TYPES)->str:
        """Uses Depth-First-Search to create code from DAG
        :param dag_node: current DAG node
        :return: the variable name the current DAG node operation created
        """

        if not isinstance(dag_node, DAGNode):
            if isinstance(dag_node, bool):
                return 'TRUE' if dag_node else 'FALSE'
            return str(dag_node)

        # If the node already has a name, then it is already defined
        # in the script, therefore reuse.
        if dag_node.daphnedsl_name != "":
            return dag_node.daphnedsl_name
        
        if dag_node._source_node is not None:
            self._dfs_dag_nodes(dag_node._source_node)
        # For each node do the dfs operation and save the variable names in `input_var_names`.
        # Get variable names of unnamed parameters.

        unnamed_input_vars = [self._dfs_dag_nodes(input_node) for input_node in dag_node.unnamed_input_nodes]

        named_input_vars = {}
        if dag_node.named_input_nodes:
            for name, input_node in dag_node.named_input_nodes.items():
                named_input_vars[name] = self._dfs_dag_nodes(input_node)
                # if isinstance(input_node, DAGNode) and input_node._output_type == OutputType.LIST:
                #     dag_node.daphnedsl_name = named_input_vars[name] + name
                #     return dag_node.daphnedsl_name

        # Check if the node gets a name after multi-returns.
        # If it has, return that name.
        if dag_node.daphnedsl_name != "":
            return dag_node.daphnedsl_name

        dag_node.daphnedsl_name = self._next_unique_var()

        if dag_node.is_python_local_data:
            self.add_input_from_python(dag_node.daphnedsl_name, dag_node)
        code_line = dag_node.code_line(
            dag_node.daphnedsl_name, unnamed_input_vars, named_input_vars)
        self.add_code(code_line)
        return dag_node.daphnedsl_name

    def _next_unique_var(self)->str:
        var_id = self._variable_counter
        self._variable_counter += 1
        return f'L{self._nested_level}_V{var_id}'