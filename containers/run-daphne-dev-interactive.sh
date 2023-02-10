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

DOCKER_IMAGE=daphneeu/daphne-dev-interactive
DAPHNE_ROOT=$PWD

LD_LIBRARY_PATH=/daphne/lib:$LD_LIBRARY_PATH
PATH=/daphne/bin:$PATH

# shellcheck disable=SC2046
docker run --hostname daphne-container -it --rm -w "$DAPHNE_ROOT" \
 -e TERM=screen-256color -e PATH -e LD_LIBRARY_PATH -e USER=$(id -n -u) -e UID=$(id -u) -e GID=$(id -g) \
 -v "$DAPHNE_ROOT:$DAPHNE_ROOT" $DOCKER_IMAGE
