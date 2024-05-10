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

from itertools import chain
from typing import Dict, Iterable, Sequence
from daphne.utils.consts import VALID_INPUT_TYPES

def create_params_string(unnamed_parameters: Iterable[str], named_parameters:Dict[str,str])->str:
    named_input_strs=(f'{v}'for (k,v) in named_parameters.items())
    return ','.join(chain(unnamed_parameters, named_input_strs))
