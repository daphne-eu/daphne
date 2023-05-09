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

echo "Use this as an example to start DAPHNE docker containers. Copy and customize for the various flavors."

DOCKER_IMAGE=daphneeu/daphne-dev
#DOCKER_TAG=latest
DOCKER_TAG=2023-05-05_cuda-12.1.1-cudnn8-devel-ubuntu20.04

# run this script from the base path of your DAPHNE source tree
DAPHNE_ROOT=$PWD
DAPHNE_ROOT_CONTAINER=/daphne

USERNAME=$(id -n -u)
GID=$(id -g)
CUDA_PATH=/usr/local/cuda
LD_LIBRARY_PATH=$CUDA_PATH/lib64:$DAPHNE_ROOT/lib:/usr/local/lib:$LD_LIBRARY_PATH
PATH=$CUDA_PATH/bin:$DAPHNE_ROOT/bin:$PATH
# uncomment to pass GPU devices to the container
DEVICE_FLAGS="--gpus all"
DEBUG_FLAGS="--cap-add=SYS_PTRACE --security-opt seccomp=unconfined"

# non-interactive: launch with PWD mounted
#docker run $DEVICE_FLAGS --user=$UID:$GID --rm -w "$DAPHNE_ROOT" -v "$DAPHNE_ROOT:$DAPHNE_ROOT" \
#    -e TERM=screen-256color -e PATH="$PATH" -e LD_LIBRARY_PATH="$LD_LIBRARY_PATH" -e USER=$USERNAME -e UID=$UID \
#    "$DOCKER_IMAGE:$DOCKER_TAG" $@

# for interactive use:
docker run $DEBUG_FLAGS $DEVICE_FLAGS -it --rm --hostname daphne-container -w $DAPHNE_ROOT_CONTAINER \
    -v "$DAPHNE_ROOT:$DAPHNE_ROOT_CONTAINER" -e GID=$GID -e TERM=screen-256color -e PATH -e LD_LIBRARY_PATH \
    -e USER=$USERNAME -e UID=$UID \
    "$DOCKER_IMAGE:$DOCKER_TAG" $@

# move this up to above the DOCKER_IMAGE line to override the entrypoint:
#    --entrypoint /daphne/containers/entrypoint-interactive.sh
