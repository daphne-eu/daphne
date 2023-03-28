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

DOCKER_IMAGE=daphneeu/daphne-dev
DAPHNE_ROOT=$PWD

LD_LIBRARY_PATH=/daphne/lib:$LD_LIBRARY_PATH
PATH=/daphne/bin:$PATH

# shellcheck disable=SC2046
# shellcheck disable=SC2068
docker run --user=$(id -u):$(id -g) --rm -w $DAPHNE_ROOT -e TERM=screen-256color -v "$DAPHNE_ROOT:$DAPHNE_ROOT" $DOCKER_IMAGE $@
