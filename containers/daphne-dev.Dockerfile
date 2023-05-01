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

ARG BASE_IMAGE=ubuntu:20.04
ARG FINAL_BASE_IMAGE=ubuntu:20.04
ARG CMAKE_VERSION=3.26.2
ARG TIMESTAMP=0

FROM ${BASE_IMAGE} as base
ARG DEBIAN_FRONTEND="noninteractive"
ARG DEBCONF_NOWARNINGS="yes"
RUN apt-get -qq -y update && apt-get -y upgrade \
    && apt-get -y --no-install-recommends install  \
    ca-certificates file git unzip wget tar \
    libomp-dev libssl-dev libxml2-dev uuid-dev zlib1g-dev \
    build-essential clang gfortran lld lldb llvm ninja-build openjdk-11-jdk-headless pkg-config \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

FROM base as with-cmake
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

FROM with-cmake as daphne-deps-compile
ARG DAPHNE_DIR=/daphne
ARG DAPHNE_REPO=https://github.com/daphne-eu/daphne.git
ARG DAPHNE_BRANCH=main
ARG TIMESTAMP
ARG CREATION_DATE=0
ARG GIT_HASH=0
LABEL "org.opencontainers.image.source"="${DAPHNE_REPO}"
LABEL "org.opencontainers.image.base.name"="${BASE_IMAGE}"
LABEL "org.opencontainers.image.version"="$TIMESTAMP"
LABEL "org.opencontainers.image.created"="${CREATION_DATE}"
LABEL "org.opencontainers.image.revision"="${GIT_HASH}"
RUN git clone --depth=1 --single-branch --branch=$DAPHNE_BRANCH $DAPHNE_REPO $DAPHNE_DIR
WORKDIR $DAPHNE_DIR
RUN ./build.sh --no-fancy --no-submodule-update --installPrefix /usr/local
RUN find /usr/local -exec file {} \; | grep -e "not stripped" | cut -d ":" -f 1 | xargs strip --strip-unneeded
RUN rm -rf $DAPHNE_DIR
RUN ldconfig
WORKDIR /

FROM ${FINAL_BASE_IMAGE} as daphne-dev
ARG DEBIAN_FRONTEND="noninteractive"
RUN apt-get -qq -y update && apt-get -y upgrade && apt-get -y --no-install-recommends install  \
    libssl1.1 libxml2-dev zlib1g-dev libtinfo-dev uuid-dev python3-numpy python3-pandas \
    build-essential ninja-build openjdk-11-jdk-headless pkg-config \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
COPY --from=daphne-deps-compile /usr/local/bin/ /usr/local/bin/
COPY --from=daphne-deps-compile /usr/local/include/ /usr/local/include/
COPY --from=daphne-deps-compile /usr/local/lib/ /usr/local/lib/
COPY --from=daphne-deps-compile /usr/local/share/ /usr/local/share/
RUN ldconfig

FROM daphne-dev as daphne-dev-interactive
ARG DEBIAN_FRONTEND="noninteractive"
RUN apt-get -qq -y update && apt-get -y upgrade && apt-get -y --no-install-recommends install  \
    vim nano rsync sudo iputils-ping virtualenv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
COPY entrypoint-interactive.sh /
ENTRYPOINT [ "/entrypoint-interactive.sh"]

FROM daphne-dev as github-action
ARG DEBIAN_FRONTEND="noninteractive"
RUN apt-get -qq -y update && apt-get -y upgrade && apt-get -y --no-install-recommends install  \
    moreutils && apt-get clean && rm -rf /var/lib/apt/lists/*
