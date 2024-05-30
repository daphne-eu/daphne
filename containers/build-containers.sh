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

# Stop immediately if any command fails.
set -e

echo "This script is primarily meant for image maintainers. Users are advised to pull from https://hub.docker.com/u/daphneeu"
echo
echo "Comment/uncomment the docker command for the image(s) to be built"
echo "If images are wrongly rebuilt from cache, add --no-cache to the docker command"
echo "For verbose output add --progress=plain to the docker build command"
echo
sleep 4

ARCH=X86-64
if [ $(arch) == 'armv*'  ] || [ $(arch) == 'aarch64' ]; then
  echo "Building for ARMv8 architecture"
  ARCH=ARMV8
fi

# read needed software versions (e.g., CUDA version,...)
source ../software-package-versions.txt

#on some installations docker can only be run with sudo
USE_SUDO=
#USE_SUDO=sudo

GIT_REPO=daphne
GIT_BRANCH="main"
GH_USER="daphne-eu"
DAPHNE_REPO_URL="https://github.com/$GH_USER/$GIT_REPO.git"

export DOCKER_BUILDKIT=1
export PROGRESS_NO_TRUNC=1

TIMESTAMP_DATE=$(date -I)
TIMESTAMP_MINUTES=$(date +"%Y-%m-%dT%H%M") # use this to add hours and minutes to the timestamp
TIMESTAMP_RFC3339=$(date --rfc-3339=seconds)
BUILD_OUTPUT_LOGFILE=docker-build-log-$TIMESTAMP_MINUTES.txt

function build_daphne() {
    $USE_SUDO docker build --progress=plain --tag "$IMAGE_REPO:$DAPHNE_TAG" --tag "$IMAGE_REPO:latest" \
        --build-arg NUM_CORES="$(nproc)" --build-arg TIMESTAMP="$TIMESTAMP_DATE" \
        --build-arg GIT_HASH="$(curl -s https://api.github.com/repos/$GH_USER/$GIT_REPO/branches/$GIT_BRANCH | \
                jq --raw-output '.commit["sha"]' -)" \
        --build-arg CMAKE_VERSION="${cmakeVersion}" \
        --build-arg BASE_IMAGE="$BASE_IMAGE" --build-arg FINAL_BASE_IMAGE="$FINAL_BASE_IMAGE"  \
        --build-arg DAPHNE_REPO=$DAPHNE_REPO_URL --build-arg DAPHNE_BRANCH=$GIT_BRANCH \
        --build-arg CREATION_DATE="$TIMESTAMP_RFC3339" --build-arg DAPHNE_BUILD_FLAGS="$DAPHNE_BUILD_FLAGS" \
        --target "$DAPHNE_TARGET" -f ./daphne"$1".Dockerfile . 2>&1 | tee "$BUILD_OUTPUT_LOGFILE"
}

#------------------------------------------------------------------------------
# Images for building dependencies and continuous integration (GitHub Actions)
#------------------------------------------------------------------------------

## testing:
#BASE_IMAGE=ubuntu:${ubuntuVersion}
#DAPHNE_TAG=$TIMESTAMP_DATE
#DAPHNE_TARGET=build-cmake
#IMAGE_REPO=local/build-cmake
##bulid deps stage
#build_daphne -deps
#exit

# this is to get the github-actions container ahead of main with 3rd party deps
#GIT_BRANCH="future-deps"
#GH_USER="corepointer"
#DAPHNE_REPO_URL="https://github.com/$GH_USER/$GIT_REPO.git"

DAPHNE_TARGET=daphne-deps
BASE_IMAGE=ubuntu:${ubuntuVersion}
DAPHNE_TAG=$TIMESTAMP_DATE_${ARCH}
IMAGE_REPO=daphneeu/$DAPHNE_TARGET
#bulid deps stage
build_daphne -deps

## build ci stage (based on deps stage)
DAPHNE_TARGET=github-action
IMAGE_REPO=daphneeu/$DAPHNE_TARGET
build_daphne -deps

## switch to main branch to build images from there (by default;only relevant if future-deps branch is used)
GIT_BRANCH="main"
GH_USER="daphne-eu"
DAPHNE_REPO_URL="https://github.com/$GH_USER/$GIT_REPO.git"

#------------------------------------------------------------------------------
# Images for DAPHNE development (BASE)
#------------------------------------------------------------------------------
DAPHNE_TARGET=daphne-dev
BASE_IMAGE=ubuntu:${ubuntuVersion}
DAPHNE_TAG=${TIMESTAMP_DATE}_${ARCH}_BASE_ubuntu${ubuntuVersion}
IMAGE_REPO=daphneeu/$DAPHNE_TARGET
build_daphne -dev

# testing new ubuntu base on dev images
ubuntuVersion=22.04
BASE_IMAGE=ubuntu:${ubuntuVersion}
DAPHNE_TAG=${TIMESTAMP_DATE}_${ARCH}_BASE_ubuntu${ubuntuVersion}
build_daphne -dev
$USE_SUDO docker tag $IMAGE_REPO:$DAPHNE_TAG daphneeu/daphne-dev:latest_${ARCH}_BASE
ubuntuVersion=20.04

#------------------------------------------------------------------------------
# Images for DAPHNE development (CUDA)
#------------------------------------------------------------------------------
DAPHNE_TARGET=daphne-dev
CUDA_TAG=${cudaVersion}-cudnn8-devel-ubuntu${ubuntuVersion}
BASE_IMAGE=nvidia/cuda:$CUDA_TAG
DAPHNE_TAG=${TIMESTAMP_DATE}_${ARCH}_CUDA_${CUDA_TAG}
IMAGE_REPO=daphneeu/$DAPHNE_TARGET
build_daphne -dev

# testing new ubuntu base on dev images
ubuntuVersion=22.04
CUDA_TAG=${cudaVersion}-cudnn8-devel-ubuntu${ubuntuVersion}
BASE_IMAGE=nvidia/cuda:$CUDA_TAG
DAPHNE_TAG=${TIMESTAMP_DATE}_${ARCH}_CUDA_${CUDA_TAG}
build_daphne -dev
$USE_SUDO docker tag $IMAGE_REPO:$DAPHNE_TAG daphneeu/daphne-dev:latest_${ARCH}_CUDA
ubuntuVersion=20.04

#-----------------------------------------------------------------------------
# Images for DAPHNE development (OneAPI)
#------------------------------------------------------------------------------
#DAPHNE_TARGET=daphne-dev
#ONEAPI_TAG=2023.1.0-devel-ubuntu${ubuntuVersion}
#BASE_IMAGE=intel/oneapi:$ONEAPI_TAG
#DAPHNE_TAG=${TIMESTAMP_DATE}_${ONEAPI_TAG}
#IMAGE_REPO=daphneeu/$DAPHNE_TARGET
#build_daphne -dev

#------------------------------------------------------------------------------
# Images for running DAPHNE
#------------------------------------------------------------------------------
DAPHNE_TARGET=daphne
BASE_IMAGE=daphneeu/daphne-deps
FINAL_BASE_IMAGE=ubuntu:${ubuntuVersion}
DAPHNE_TAG=${TIMESTAMP_DATE}_${ARCH}_BASE_ubuntu${ubuntuVersion}
IMAGE_REPO=daphneeu/$DAPHNE_TARGET
DAPHNE_BUILD_FLAGS="--mpi"
build_daphne
$USE_SUDO docker tag $IMAGE_REPO:$DAPHNE_TAG daphneeu/daphne:latest_${ARCH}_BASE

#-----------------------------------------------------------------------------
# Images for running DAPHNE (CUDA)
#------------------------------------------------------------------------------
DAPHNE_TARGET=daphne
CUDA_TAG=${cudaVersion}-cudnn8-runtime-ubuntu${ubuntuVersion}
DAPHNE_TAG=${TIMESTAMP_DATE}_${ARCH}_CUDA_${CUDA_TAG}
IMAGE_REPO=daphneeu/$DAPHNE_TARGET
BASE_IMAGE=daphneeu/daphne-dev
FINAL_BASE_IMAGE=nvidia/cuda:$CUDA_TAG
DAPHNE_BUILD_FLAGS="--mpi --cuda"
build_daphne
$USE_SUDO docker tag $IMAGE_REPO:$DAPHNE_TAG daphneeu/daphne:latest_${ARCH}_CUDA

set +e
