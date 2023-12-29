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


# defaults:
ARG BASE_IMAGE=ubuntu:20.04
ARG CMAKE_VERSION=3.28.1
ARG DEBIAN_FRONTEND="noninteractive"
ARG DEBCONF_NOWARNINGS="yes"
ARG DAPHNE_DIR=/daphne
ARG DAPHNE_REPO=https://github.com/daphne-eu/daphne.git
ARG DAPHNE_BRANCH=main
ARG TIMESTAMP=0
ARG CREATION_DATE=0
ARG GIT_HASH=0

FROM ${BASE_IMAGE} as base
ARG DEBIAN_FRONTEND
ARG DEBCONF_NOWARNINGS
RUN apt-get -qq -y update && apt-get -y upgrade \
    && apt-get -y --no-install-recommends install  \
    ca-certificates file git openssh-client unzip wget tar \
    libomp-dev  libpfm4-dev libssl-dev libxml2-dev uuid-dev zlib1g-dev \
    build-essential clang gfortran lld llvm llvm-10-tools ninja-build openjdk-11-jdk-headless pkg-config python3 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

FROM base as build-cmake
ARG NUM_CORES=4
ARG CMAKE_VERSION
ARG BUILD_DIR=/build-cmake
WORKDIR $BUILD_DIR
RUN wget -qO- https://github.com/Kitware/CMake/releases/download/v$CMAKE_VERSION/cmake-$CMAKE_VERSION.tar.gz | tar xzf \
    - --strip-components=1
RUN ./bootstrap --parallel=$NUM_CORES --generator=Ninja --no-qt-gui --prefix=/usr/local
RUN ninja
RUN strip --strip-unneeded bin/*
RUN ninja install
WORKDIR /
RUN rm -rf ${BUILD_DIR}

FROM build-cmake as build
ARG DAPHNE_DIR=/daphne
ARG DAPHNE_REPO=https://github.com/daphne-eu/daphne.git
ARG DAPHNE_BRANCH=main
RUN git clone --depth=1 --single-branch --branch=$DAPHNE_BRANCH $DAPHNE_REPO $DAPHNE_DIR
WORKDIR $DAPHNE_DIR
RUN ./build.sh --no-fancy --no-submodule-update --installPrefix /usr/local
RUN find /usr/local -exec file {} \; | grep -e "not stripped" | cut -d ":" -f 1 | xargs strip --strip-unneeded
RUN rm -rf $DAPHNE_DIR
RUN ldconfig
WORKDIR /

FROM base as daphne-deps
ARG DAPHNE_REPO
ARG DAPHNE_BRANCH
ARG TIMESTAMP
ARG CREATION_DATE
ARG GIT_HASH
LABEL "org.opencontainers.image.source"="${DAPHNE_REPO}"
LABEL "org.opencontainers.image.base.name"="${BASE_IMAGE}"
LABEL "org.opencontainers.image.version"="branch_${DAPHNE_BRANCH}_from_${TIMESTAMP}"
LABEL "org.opencontainers.image.created"="${CREATION_DATE}"
LABEL "org.opencontainers.image.revision"="${GIT_HASH}"
COPY --from=build /usr/local/bin/ /usr/local/bin/
COPY --from=build /usr/local/include/ /usr/local/include/
COPY --from=build /usr/local/lib/ /usr/local/lib/
COPY --from=build /usr/local/share/ /usr/local/share/
RUN ldconfig

FROM daphneeu/daphne-deps as github-action
RUN apt-get -qq -y update && apt-get -y upgrade && apt-get -y --no-install-recommends install  \
    moreutils python3-numpy python3-pandas ccache python3-pip python3-networkx python3-dev graphviz-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
