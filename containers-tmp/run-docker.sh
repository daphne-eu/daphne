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

# Run this script on the host from the base path of your DAPHNE source tree
# to start the DAPHNE container.

DOCKER_IMAGE=daphneeu/daphne-dev
DOCKER_TAG=latest

DAPHNE_ROOT=$PWD
DAPHNE_ROOT_CONTAINER=/daphne

USERNAME=$(id -n -u)
GID=$(id -g)
LD_LIBRARY_PATH=$DAPHNE_ROOT/lib:/usr/local/lib:$LD_LIBRARY_PATH
PATH=$DAPHNE_ROOT/bin:$PATH
DEBUG_FLAGS="--cap-add=SYS_PTRACE --security-opt seccomp=unconfined"

docker run $DEBUG_FLAGS -it --rm --hostname daphne-container -w $DAPHNE_ROOT_CONTAINER \
    -v "$DAPHNE_ROOT:$DAPHNE_ROOT_CONTAINER" -e GID=$GID -e TERM=screen-256color -e PATH -e LD_LIBRARY_PATH \
    -e USER=$USERNAME -e UID=$UID \
    --entrypoint containers-tmp/entrypoint.sh \
    "$DOCKER_IMAGE:$DOCKER_TAG" bash