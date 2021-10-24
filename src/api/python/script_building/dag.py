from abc import ABC
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Dict, Sequence, Union, Optional

class OutputType(Enum):
        MATRIX = auto()
        LIST = auto()
        NONE = auto()
        DOUBLE = auto()
        

class DAGNode(ABC):
    _unnamed_input_nodes: Sequence[Union['DAGNode', str, int, float, bool]]
    _named_input_nodes:Dict[str, Union['DAGNode', str, int, float, bool]]
    _named_output_nodes:Dict[str, Union['DAGNode', str, int, float, bool]]
    _source_node: Optional["DAGNode"]
    _output_type: OutputType
    _script: Optional["DSLScript"]
    _is_python_local_data: bool
    _dsl_name: str

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
    def script_str(self):
        return self._script.dml_script

    @property
    def dsl_name(self):
        return self._dsl_name

    @dsl_name.setter
    def dsl_name(self, value):
        self._dsl_name = value