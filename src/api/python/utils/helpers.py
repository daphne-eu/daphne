from itertools import chain
from typing import Dict, Iterable, Sequence
from api.python.utils.consts import VALID_INPUT_TYPES

def create_params_string(unnamed_parameters: Iterable[str], named_parameters:Dict[str,str])->str:
    named_input_strs=(f'{v}'for (k,v) in named_parameters.items())
    return ','.join(chain(unnamed_parameters, named_input_strs))
