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


# This Dockerfile provides an image based on the basic Daphne compilation environment
# and adds some useful packages for an interactive session

FROM daphneeu/daphne-dev
ARG DEBIAN_FRONTEND="noninteractive"
RUN apt-get -qq -y update && apt-get -y upgrade && apt-get -y --no-install-recommends install  \
    vim nano openssh-client rsync sudo iputils-ping virtualenv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
COPY entrypoint-interactive.sh /
ENTRYPOINT [ "/entrypoint-interactive.sh"]