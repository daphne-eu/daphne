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

from abc import ABC
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Dict, Sequence, Union, Optional

class OutputType(Enum):
    MATRIX = auto()
    NONE = auto()
    SCALAR = auto()
    FRAME = auto()
    MULTI_RETURN = auto()

class DAGNode(ABC):
    _unnamed_input_nodes: Sequence[Union['DAGNode', str, int, float, bool]]
    _named_input_nodes:Dict[str, Union['DAGNode', str, int, float, bool]]
    _named_output_nodes:Dict[str, Union['DAGNode', str, int, float, bool]]
    _source_node: Optional["DAGNode"]
    _output_type: OutputType
    _script: Optional["DaphneDSLScript"]
    _is_python_local_data: bool
    _daphnedsl_name: str

    def compute() -> Any:
        raise NotImplementedError

    def code_line(self, var_name:str, unnamed_input_vars:Sequence[str],named_input_vars:Dict[str,str])->str:
        raise NotImplementedError

    @property
    def unnamed_input_nodes(self):
        return self._unnamed_input_nodes

    @property
    def named_input_nodes(self):
        return self._named_input_nodes

    @property
    def named_output_nodes(self):
        return self._named_output_nodes

    @property
    def is_python_local_data(self):
        return self._is_python_local_data

    @property
    def output_type(self):
        return self._output_type

    @property
    def script(self):
        return self._script

    @property
    def daphnedsl_name(self):
        return self._daphnedsl_name

    @daphnedsl_name.setter
    def daphnedsl_name(self, value):
        self._daphnedsl_name = value