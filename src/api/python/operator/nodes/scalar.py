from typing import (TYPE_CHECKING, Dict, Iterable, Optional, Sequence, Tuple,
                    Union)
from api.python.script_building.dag import OutputType
from api.python.utils.consts import VALID_INPUT_TYPES
import numpy as np
from typing import Iterable

from api.python.operator.operation_node import OperationNode


class Scalar(OperationNode):
    __assign: bool

    def __init__(self, operation: str,
                 unnamed_input_nodes: Iterable[VALID_INPUT_TYPES] = None,
                 named_input_nodes: Dict[str, VALID_INPUT_TYPES] = None,
                 output_type: OutputType = OutputType.DOUBLE,
                 assign: bool = False) -> 'Scalar':
        self.__assign = assign
        super().__init__( operation, unnamed_input_nodes=unnamed_input_nodes,
                         named_input_nodes=named_input_nodes, output_type=output_type)

    def code_line(self, var_name: str, unnamed_input_vars: Sequence[str],
                  named_input_vars: Dict[str, str]) -> str:
        if self.__assign:
            return f'{var_name}={self.operation};'
        else:
            return super().code_line(var_name, unnamed_input_vars, named_input_vars)

    def compute(self) -> Union[np.array]:
        return super().compute()