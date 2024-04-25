#!/usr/bin/env bash

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

DAPHNE_ROOT=$PWD
export LD_LIBRARY_PATH=$DAPHNE_ROOT/lib:$DAPHNE_ROOT/thirdparty/installed/lib:$LD_LIBRARY_PATH
export PYTHONPATH="$PYTHONPATH:$PWD/src/api/python/"
export DAPHNELIB_DIR_PATH=$DAPHNE_ROOT/lib

# Silence TensorFlow warnings in DaphneLib.
export TF_CPP_MIN_LOG_LEVEL=3

python3 $@
