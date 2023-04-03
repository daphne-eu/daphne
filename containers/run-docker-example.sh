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
#DOCKER_IMAGE=daphneeu/daphne-dev-interactive
DOCKER_TAG=latest
#DOCKER_TAG=cuda-12.0.1-cudnn8-devel-ubuntu20.04_2023-04-03
DAPHNE_ROOT=$PWD

CUDA_PATH=/usr/local/cuda
LD_LIBRARY_PATH=$CUDA_PATH/lib64:$DAPHNE_ROOT/lib:$LD_LIBRARY_PATH
PATH=$CUDA_PATH/bin:$DAPHNE_ROOT/bin:$PATH
# uncomment to pass GPU devices to the container
#DEVICE_FLAGS=--gpus all

# shellcheck disable=SC2046
# shellcheck disable=SC2068
docker run "$DEVICE_FLAGS" --user=$(id -u):$(id -g) --rm -w "$DAPHNE_ROOT" -v "$DAPHNE_ROOT:$DAPHNE_ROOT" \
    -e TERM=screen-256color -e PATH="$PATH" -e LD_LIBRARY_PATH="$LD_LIBRARY_PATH" -e USER=$(id -n -u) -e UID=$(id -u) \
    "$DOCKER_IMAGE:$DOCKER_TAG" $@
