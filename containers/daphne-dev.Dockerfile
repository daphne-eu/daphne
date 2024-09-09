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
#ARG FINAL_BASE_IMAGE=ubuntu:20.04
ARG CMAKE_VERSION=3.29.3
ARG TIMESTAMP=0
ARG TZ=Etc/UTC

FROM ${BASE_IMAGE} AS daphne-dev
ARG DEBIAN_FRONTEND="noninteractive"
ARG TZ
RUN apt-get -qq -y update && apt-get -y upgrade && apt-get -y --no-install-recommends install  \
    ca-certificates file git openssh-client unzip wget tar \
    libomp-dev  libpfm4-dev libssl-dev libxml2-dev uuid-dev zlib1g-dev \
    build-essential clang gfortran lld llvm llvm-18-tools ninja-build openjdk-11-jdk-headless pkg-config python3-numpy python3-pandas \
    vim nano rsync sudo iputils-ping virtualenv openssh-server iproute2 git htop gdb lldb lld gpg-agent net-tools \
    software-properties-common ca-certificates file unzip wget tar zstd \
    ccache python3-pip python3-networkx python3-dev graphviz-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
COPY --from=daphneeu/daphne-deps /usr/local/bin/ /usr/local/bin/
COPY --from=daphneeu/daphne-deps /usr/local/include/ /usr/local/include/
COPY --from=daphneeu/daphne-deps /usr/local/lib/ /usr/local/lib/
COPY --from=daphneeu/daphne-deps /usr/local/share/ /usr/local/share/
RUN ldconfig
# this is a temporary workaround to make the lit code (from the llvm-*-tools package) available to some pre-Ubuntu24 \
# test cases when run locally in the dev container
RUN ln -s /usr/lib/llvm-18 /usr/lib/llvm-10
RUN ln -fs /usr/share/zoneinfo/$TZ /etc/localtime
COPY entrypoint-interactive.sh /
RUN mkdir -p /var/run/sshd
EXPOSE 22
ENTRYPOINT [ "/entrypoint-interactive.sh"]
