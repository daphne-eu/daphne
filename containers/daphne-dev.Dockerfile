# syntax=docker/dockerfile:1

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


# This Dockerfile provides a basic DAPHNE compilation environment with all
# third party dependencies precompiled (use ''./build.sh --no-deps --installPrefix /usr/local'' to compile DAPHNE)

FROM ubuntu:20.04 as base
ARG DEBIAN_FRONTEND="noninteractive"
ARG DEBCONF_NOWARNINGS="yes"
RUN apt-get -qq -y update && apt-get -y upgrade \
    && apt-get -y --no-install-recommends install  \
    ca-certificates git unzip wget tar \
    libomp-dev libssl-dev libxml2-dev uuid-dev zlib1g-dev \
    build-essential clang gfortran lld lldb llvm ninja-build openjdk-11-jdk-headless pkg-config \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
ARG NUM_CORES=4
ARG CMAKE_VERSION=3.25.2
ARG BUILD_DIR=/build-cmake
WORKDIR $BUILD_DIR
RUN wget -qO- https://github.com/Kitware/CMake/releases/download/v$CMAKE_VERSION/cmake-$CMAKE_VERSION.tar.gz | tar xzf \
    - --strip-components=1
RUN ./bootstrap --parallel=$NUM_CORES --generator=Ninja --no-qt-gui --prefix=/usr/local
RUN ninja install
WORKDIR /
RUN rm -rf ${BUILD_DIR}

FROM base as with-deps
ARG DAPHNE_DIR=/usr/local/src/daphne
ARG DAPHNE_REPO=https://github.com/corepointer/daphne.git
ARG DAPHNE_BRANCH=docker-improvements
RUN git clone --depth=1 --single-branch --branch=$DAPHNE_BRANCH $DAPHNE_REPO $DAPHNE_DIR
WORKDIR $DAPHNE_DIR
RUN ./build.sh -nf --no-submodule-update --installPrefix /usr/local
WORKDIR /
RUN rm -rf $DAPHNE_DIR

FROM ubuntu:20.04 as daphne-dev
ARG DEBIAN_FRONTEND="noninteractive"
RUN apt-get -qq -y update && apt-get -y upgrade \
    && apt-get -y --no-install-recommends install  \
    ca-certificates git unzip wget tar zstd moreutils \
    libomp-dev libssl-dev libxml2-dev uuid-dev zlib1g-dev \
    build-essential clang gdb gdbserver gfortran lld lldb llvm ninja-build openjdk-11-jdk-headless pkg-config \
    python3-numpy && apt-get clean && rm -rf /var/lib/apt/lists/*
COPY --from=with-deps /usr/local /usr/local
