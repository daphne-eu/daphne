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
echo "Add sudo to docker invocation if needed in your setup"

ARCH=X86-64
DOCKER_IMAGE=daphneeu/daphne-dev
DOCKER_TAG=latest_${ARCH}_BASE
#DOCKER_TAG=latest_${ARCH}_CUDA

#on some installations docker can only be run with sudo
USE_SUDO=
#USE_SUDO=sudo

# run this script from the base path of your DAPHNE source tree
DAPHNE_ROOT=$PWD

# the directory used *inside* the container to bind-mount your source directory
DAPHNE_ROOT_CONTAINER=/daphne

# user info to set up your user inside the container (to avoid creating files that then belong to the root user)
USERNAME=$(id -n -u)
GID=$(id -g)

# some environment setup
CUDA_PATH=/usr/local/cuda
LD_LIBRARY_PATH=$CUDA_PATH/lib64:$DAPHNE_ROOT/lib:/usr/local/lib:$LD_LIBRARY_PATH
# temporarily adding this NSight Systems path
PATH=/opt/nvidia/nsight-compute/2023.2.2/host/target-linux-x64:$CUDA_PATH/bin:$DAPHNE_ROOT/bin:$PATH

# uncomment the appropriate to pass GPU devices to the container (goes hand in hand with DOCKER_TAG)
DEVICE_FLAGS=""
#DEVICE_FLAGS="--gpus all"

# this might be needed if a debugging session is run in the container
DEBUG_FLAGS=""
#DEBUG_FLAGS="--cap-add=SYS_PTRACE --security-opt seccomp=unconfined"

# set bash as the default command if none is provided
command=$*
if [ "$#" -eq 0 ]; then
    command=bash
fi

# non-interactive: launch with PWD mounted
#docker run $DEVICE_FLAGS --user=$UID:$GID --rm -w "$DAPHNE_ROOT" -v "$DAPHNE_ROOT:$DAPHNE_ROOT" \
#    -e TERM=screen-256color -e PATH="$PATH" -e LD_LIBRARY_PATH="$LD_LIBRARY_PATH" -e USER=$USERNAME -e UID=$UID \
#    "$DOCKER_IMAGE:$DOCKER_TAG" $@

# for interactive use:
$USE_SUDO docker run $DEBUG_FLAGS $DEVICE_FLAGS -it --rm --hostname daphne-container -w $DAPHNE_ROOT_CONTAINER \
    -v "$DAPHNE_ROOT:$DAPHNE_ROOT_CONTAINER" -e GID=$GID -e TERM=screen-256color -e PATH -e LD_LIBRARY_PATH \
    -e USER=$USERNAME -e UID=$UID \
    "$DOCKER_IMAGE:$DOCKER_TAG" $command

# move this up to above the DOCKER_IMAGE line to override the entrypoint:
#    --entrypoint /daphne/containers/entrypoint-interactive.sh
