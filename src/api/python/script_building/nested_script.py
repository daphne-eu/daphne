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
# Modifications Copyright 2023 The DAPHNE Consortium
#
# -------------------------------------------------------------

from api.python.script_building.dag import DAGNode, OutputType
from api.python.operator.operation_node import OperationNode
from api.python.utils.consts import VALID_INPUT_TYPES, TMP_PATH, PROTOTYPE_PATH
from api.python.utils.daphnelib import DaphneLib
from api.python.script_building.script import DaphneDSLScript

import ctypes
import os
from typing import List, Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    # to avoid cyclic dependencies during runtime
    from context.daphne_context import DaphneContext
    
class NestedDaphneDSLScript(DaphneDSLScript):
    var_prefix: str
    def __init__(self, context, nested_level=1, var_prefix='L') -> None:
        super().__init__(context)
        self._nested_level = nested_level
        self.var_prefix = var_prefix

    def build_code(self, dag_tuple: Tuple['DAGNode']):
        names = list()
        for node in dag_tuple:
            names.append(self._dfs_dag_nodes(node))
        return names

    def execute(self):
        raise TypeError("NestedDaphneDSLScript can not be executed!")
    
    def _dfs_dag_nodes(self, dag_node: VALID_INPUT_TYPES)->str:
        if isinstance(dag_node, OperationNode) and dag_node._source_node is not None:
            dag_node._source_node.nested_level += 1
        return super()._dfs_dag_nodes(dag_node)

    def _next_unique_var(self)->str:
        var_id = self._variable_counter
        self._variable_counter += 1
        return f'{self.var_prefix}{self._nested_level}_V{var_id}'