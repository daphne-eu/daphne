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

echo "This script is primarily meant for image maintainers. Users are advised to pull from https://hub.docker.com/u/daphneeu"
echo
echo "Comment/uncomment the docker command for the image(s) to be built"
echo "If images are wrongly rebuilt from cache, add --no-cache to the docker command"
echo "For verbose output add --progress=plain to the docker build command"

GIT_REPO=daphne
GIT_BRANCH="docker-cuda"
GH_USER="corepointer"
DAPHNE_REPO_URL="https://github.com/$GH_USER/$GIT_REPO.git"

export DOCKER_BUILDKIT=1
export PROGRESS_NO_TRUNC=1

TIMESTAMP_DATE=$(date -I)
TIMESTAMP_MINUTES=$(date +"%Y-%m-%dT%H%M") # use this to add hours and minutes to the timestamp
TIMESTAMP_RFC3339=$(date --rfc-3339=seconds)
BUILD_OUTPUT_LOGFILE=docker-build-log-$TIMESTAMP_MINUTES.txt

function build_daphne() {
    docker build --progress=plain --tag "$IMAGE_REPO:$DAPHNE_TAG" --tag "$IMAGE_REPO:latest" \
        --build-arg NUM_CORES="$(nproc)" --build-arg TIMESTAMP="$TIMESTAMP_DATE" \
        --build-arg GIT_HASH="$(curl -s https://api.github.com/repos/$GH_USER/$GIT_REPO/branches/$GIT_BRANCH | \
                jq --raw-output '.commit["sha"]' -)" \
        --build-arg BASE_IMAGE="$BASE_IMAGE" --build-arg FINAL_BASE_IMAGE="$FINAL_BASE_IMAGE"  \
        --build-arg DAPHNE_REPO=$DAPHNE_REPO_URL --build-arg DAPHNE_BRANCH=$GIT_BRANCH \
        --build-arg CREATION_DATE="$TIMESTAMP_RFC3339" --build-arg DAPHNE_BUILD_FLAGS="$DAPHNE_BUILD_FLAGS" \
        --target "$DAPHNE_TARGET" -f ./daphne"$1".Dockerfile . 2>&1 | tee "$BUILD_OUTPUT_LOGFILE"
}

#------------------------------------------------------------------------------
# Images for continuous integration (GitHub Actions)
#------------------------------------------------------------------------------
#BASE_IMAGE=ubuntu:20.04
#FINAL_BASE_IMAGE=ubuntu:20.04
#DAPHNE_TARGET=github-action
#DAPHNE_TAG=$TIMESTAMP_DATE
#IMAGE_REPO=daphneeu/$DAPHNE_TARGET
#build_daphne -dev

#------------------------------------------------------------------------------
# Images for DAPHNE development
#------------------------------------------------------------------------------
#BASE_IMAGE=ubuntu:20.04
#DAPHNE_TARGET=daphne-dev
#DAPHNE_TAG=$TIMESTAMP_DATE
#IMAGE_REPO=daphneeu/$DAPHNE_TARGET
#build_daphne -dev

#------------------------------------------------------------------------------
# Images for running DAPHNE
#------------------------------------------------------------------------------
#BASE_IMAGE=daphneeu/daphne-deps-compile
#FINAL_BASE_IMAGE=ubuntu:20.04
#DAPHNE_TARGET=daphne
#DAPHNE_TAG=ubuntu20.04_$TIMESTAMP_DATE
#IMAGE_REPO=daphneeu/$DAPHNE_TARGET
#DAPHNE_BUILD_FLAGS="--mpi"
#build_daphne

#------------------------------------------------------------------------------
# Images for DAPHNE development (CUDA)
#------------------------------------------------------------------------------
#DAPHNE_TARGET=daphne-dev
#CUDA_TAG=12.0.1-cudnn8-devel-ubuntu20.04
#DAPHNE_TAG=${CUDA_TAG}_$TIMESTAMP_DATE
#IMAGE_REPO=daphneeu/$DAPHNE_TARGET-cuda
#FINAL_BASE_IMAGE=nvidia/cuda:$CUDA_TAG
#BASE_IMAGE=ubuntu:20.04
#build_daphne -dev

#-----------------------------------------------------------------------------
# Images for running DAPHNE (CUDA)
#------------------------------------------------------------------------------
#DAPHNE_TARGET=daphne
#CUDA_TAG=12.0.1-cudnn8-runtime-ubuntu20.04
#DAPHNE_TAG=${CUDA_TAG}_$TIMESTAMP_DATE
#IMAGE_REPO=daphneeu/$DAPHNE_TARGET-cuda
#BASE_IMAGE=daphneeu/daphne-dev-cuda
#FINAL_BASE_IMAGE=nvidia/cuda:$CUDA_TAG
#DAPHNE_BUILD_FLAGS="--mpi --cuda"
#build_daphne

#-----------------------------------------------------------------------------
# Images for DAPHNE development (OneAPI)
#------------------------------------------------------------------------------
#DAPHNE_TARGET=daphne-dev-interactive
#ONEAPI_TAG=2023.0.0-devel-ubuntu22.04
#DAPHNE_TAG=${ONEAPI_TAG}_$TIMESTAMP_DATE
#IMAGE_REPO=daphneeu/$DAPHNE_TARGET-oneapi
#BASE_IMAGE=ubuntu:20.04
#FINAL_BASE_IMAGE=intel/oneapi:$ONEAPI_TAG
#build_daphne -dev
