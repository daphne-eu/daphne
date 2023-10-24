# Copyright 2023 The DAPHNE Consortium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Callable, get_type_hints
from api.python.operator.operation_node import OperationNode

def get_outer_scope_variables(function: Callable):
    free_vars = inspect.getclosurevars(function).nonlocals
    globals = inspect.getclosurevars(function).globals
    key_to_delete = list()
    for name, obj in globals.items():
        if not isinstance(obj, OperationNode):
            key_to_delete.append(name)
    for key in key_to_delete:
        del globals[key]

    return {**free_vars, **globals}

def get_number_argument(function: Callable):
    return function.__code__.co_argcount

def get_argument_types(function: Callable):
    arguments = inspect.signature(function).parameters
    arguments_type_hints = get_type_hints(function)
    return {key: arguments_type_hints.get(key) for key in arguments.keys()}
