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

ARG BASE_IMAGE=daphneeu/daphne-dev
ARG FINAL_BASE_IMAGE=ubuntu:20.04
ARG DAPHNE_DIR=/daphne
ARG DAPHNE_REPO=https://github.com/daphne-eu/daphne.git
ARG DAPHNE_BRANCH=main
ARG TIMESTAMP=0
ARG CREATION_DATE=0
ARG GIT_HASH=0
ARG TZ=Etc/UTC

FROM ${BASE_IMAGE} as daphne-build
ARG DAPHNE_DIR
ARG DAPHNE_REPO
ARG DAPHNE_BRANCH
ARG TIMESTAMP
ARG CREATION_DATE
ARG GIT_HASH
ARG DAPHNE_BUILD_FLAGS
ARG TZ
LABEL "org.opencontainers.image.revision"="${DAPHNE_REPO}@${DAPHNE_BRANCH}"
LABEL "org.opencontainers.image.base.name"="$BASE_IMAGE"
LABEL "org.opencontainers.image.version"="$TIMESTAMP"
RUN echo Timestamp: $TIMESTAMP
RUN echo $TIMESTAMP
RUN echo ${DAPHNE_REPO}
RUN echo $DAPHNE_BRANCH
RUN echo $DAPHNE_DIR 
RUN git clone --depth=1 --single-branch --branch=$DAPHNE_BRANCH $DAPHNE_REPO $DAPHNE_DIR
WORKDIR $DAPHNE_DIR
RUN ldconfig
RUN ln -fs /usr/share/zoneinfo/$TZ /etc/localtime
RUN ./build.sh --no-fancy --no-deps --no-submodule-update --installPrefix /usr/local $DAPHNE_BUILD_FLAGS
RUN strip --strip-unneeded bin/*
RUN strip --strip-unneeded lib/*\.so*
WORKDIR /

FROM ${FINAL_BASE_IMAGE} as daphne
ARG DAPHNE_DIR
ARG DAPHNE_REPO
ARG DAPHNE_BRANCH
ARG TIMESTAMP
ARG CREATION_DATE
ARG GIT_HASH
ARG TZ
LABEL "org.opencontainers.image.source"="${DAPHNE_REPO}"
LABEL "org.opencontainers.image.base.name"="${BASE_IMAGE}"
LABEL "org.opencontainers.image.version"="$TIMESTAMP"
LABEL "org.opencontainers.image.created"="${CREATION_DATE}"
LABEL "org.opencontainers.image.revision"="${GIT_HASH}"
RUN apt-get -qq -y update && apt-get -y upgrade && apt-get -y --no-install-recommends install  \
    libtinfo6 openssl zlib1g python3-numpy python3-pandas libxml2 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
COPY --from=daphne-build $DAPHNE_DIR/bin/* /usr/local/bin
COPY --from=daphne-build $DAPHNE_DIR/lib/* /usr/local/lib
COPY --from=daphne-build /usr/local/lib/lib*.so* /usr/local/lib
RUN ldconfig
RUN ln -fs /usr/share/zoneinfo/$TZ /etc/localtime
ENTRYPOINT ["/usr/local/bin/daphne"]
CMD ["--help"]
