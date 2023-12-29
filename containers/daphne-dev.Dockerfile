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
ARG CMAKE_VERSION=3.28.1
ARG TIMESTAMP=0

#FROM ${BASE_IMAGE} as base
#ARG DEBIAN_FRONTEND="noninteractive"
#ARG DEBCONF_NOWARNINGS="yes"
#RUN apt-get -qq -y update && apt-get -y upgrade \
#    && apt-get -y --no-install-recommends install  \
#    ca-certificates file git unzip wget tar \
#    libomp-dev libssl-dev libxml2-dev uuid-dev zlib1g-dev \
#    build-essential clang gfortran lld llvm ninja-build openjdk-11-jdk-headless pkg-config \
#    && apt-get clean && rm -rf /var/lib/apt/lists/*
#
#FROM base as with-cmake
#ARG NUM_CORES=4
#ARG CMAKE_VERSION
#ARG BUILD_DIR=/build-cmake
#WORKDIR $BUILD_DIR
#RUN wget -qO- https://github.com/Kitware/CMake/releases/download/v$CMAKE_VERSION/cmake-$CMAKE_VERSION.tar.gz | tar xzf \
#    - --strip-components=1
#RUN ./bootstrap --parallel=$NUM_CORES --generator=Ninja --no-qt-gui --prefix=/usr/local
#RUN ninja
#RUN strip --strip-unneeded bin/*
#RUN ninja install
#WORKDIR /
#RUN rm -rf ${BUILD_DIR}
#
#FROM with-cmake as daphne-deps-compile
#ARG DAPHNE_DIR=/daphne
#ARG DAPHNE_REPO=https://github.com/daphne-eu/daphne.git
#ARG DAPHNE_BRANCH=main
#ARG TIMESTAMP
#ARG CREATION_DATE=0
#ARG GIT_HASH=0
#LABEL "org.opencontainers.image.source"="${DAPHNE_REPO}"
#LABEL "org.opencontainers.image.base.name"="${BASE_IMAGE}"
#LABEL "org.opencontainers.image.version"="$TIMESTAMP"
#LABEL "org.opencontainers.image.created"="${CREATION_DATE}"
#LABEL "org.opencontainers.image.revision"="${GIT_HASH}"
#RUN git clone --depth=1 --single-branch --branch=$DAPHNE_BRANCH $DAPHNE_REPO $DAPHNE_DIR
#WORKDIR $DAPHNE_DIR
#RUN ./build.sh --no-fancy --no-submodule-update --installPrefix /usr/local
#RUN find /usr/local -exec file {} \; | grep -e "not stripped" | cut -d ":" -f 1 | xargs strip --strip-unneeded
#RUN rm -rf $DAPHNE_DIR
#RUN ldconfig
#WORKDIR /

FROM ${BASE_IMAGE} as daphne-dev
ARG DEBIAN_FRONTEND="noninteractive"
RUN apt-get -qq -y update && apt-get -y upgrade && apt-get -y --no-install-recommends install  \
    ca-certificates file git openssh-client unzip wget tar \
    libomp-dev  libpfm4-dev libssl-dev libxml2-dev uuid-dev zlib1g-dev \
    build-essential clang gfortran lld llvm ninja-build openjdk-11-jdk-headless pkg-config python3-numpy python3-pandas \
    vim nano rsync sudo iputils-ping virtualenv openssh-server iproute2 git htop gdb lldb lld gpg-agent net-tools \
    software-properties-common ca-certificates file unzip wget tar zstd \
    ccache python3-pip python3-networkx python3-dev graphviz-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
COPY --from=daphneeu/daphne-deps /usr/local/bin/ /usr/local/bin/
COPY --from=daphneeu/daphne-deps /usr/local/include/ /usr/local/include/
COPY --from=daphneeu/daphne-deps /usr/local/lib/ /usr/local/lib/
COPY --from=daphneeu/daphne-deps /usr/local/share/ /usr/local/share/
RUN ldconfig
#RUN useradd -rm -d /home/daphnedev -s /bin/bash -g users -G sudo -u 2000 daphnedev
#RUN mkdir -p /home/daphnedev/.ssh
#RUN chmod 700 /home/daphnedev/.ssh
COPY entrypoint-interactive.sh /
RUN mkdir -p /var/run/sshd
EXPOSE 22
ENTRYPOINT [ "/entrypoint-interactive.sh"]


## Include OneAPI:
#COPY --from=intel/oneapi-basekit:2023.1.0-devel-ubuntu20.04 /opt/intel /opt/
## Directives from https://github.com/intel/oneapi-containers/blob/master/images/docker/basekit/Dockerfile.ubuntu-20.04
#ENV LANG=C.UTF-8
#ENV ACL_BOARD_VENDOR_PATH='/opt/Intel/OpenCLFPGA/oneAPI/Boards'
#ENV ADVISOR_2023_DIR='/opt/intel/oneapi/advisor/2023.1.0'
#ENV APM='/opt/intel/oneapi/advisor/2023.1.0/perfmodels'
#ENV CCL_CONFIGURATION='cpu_gpu_dpcpp'
#ENV CCL_ROOT='/opt/intel/oneapi/ccl/2021.9.0'
#ENV CLASSPATH='/opt/intel/oneapi/mpi/2021.9.0//lib/mpi.jar:/opt/intel/oneapi/dal/2023.1.0/lib/onedal.jar'
#ENV CMAKE_PREFIX_PATH='/opt/intel/oneapi/tbb/2021.9.0/env/..:/opt/intel/oneapi/dnnl/2023.1.0/cpu_dpcpp_gpu_dpcpp/../lib/cmake:/opt/intel/oneapi/dal/2023.1.0:/opt/intel/oneapi/compiler/2023.1.0/linux/IntelDPCPP:/opt/intel/oneapi/ccl/2021.9.0/lib/cmake/oneCCL'
#ENV CMPLR_ROOT='/opt/intel/oneapi/compiler/2023.1.0'
#ENV CPATH='/opt/intel/oneapi/tbb/2021.9.0/env/../include:/opt/intel/oneapi/mpi/2021.9.0//include:/opt/intel/oneapi/mkl/2023.1.0/include:/opt/intel/oneapi/ippcp/2021.7.0/include:/opt/intel/oneapi/ipp/2021.8.0/include:/opt/intel/oneapi/dpl/2022.1.0/linux/include:/opt/intel/oneapi/dnnl/2023.1.0/cpu_dpcpp_gpu_dpcpp/include:/opt/intel/oneapi/dev-utilities/2021.9.0/include:/opt/intel/oneapi/dal/2023.1.0/include:/opt/intel/oneapi/ccl/2021.9.0/include/cpu_gpu_dpcpp'
#ENV DAALROOT='/opt/intel/oneapi/dal/2023.1.0'
#ENV DALROOT='/opt/intel/oneapi/dal/2023.1.0'
#ENV DAL_MAJOR_BINARY='1'
#ENV DAL_MINOR_BINARY='1'
#ENV DIAGUTIL_PATH='/opt/intel/oneapi/vtune/2023.1.0/sys_check/vtune_sys_check.py:/opt/intel/oneapi/debugger/2023.1.0/sys_check/debugger_sys_check.py:/opt/intel/oneapi/compiler/2023.1.0/sys_check/sys_check.sh:/opt/intel/oneapi/advisor/2023.1.0/sys_check/advisor_sys_check.py:'
#ENV DNNLROOT='/opt/intel/oneapi/dnnl/2023.1.0/cpu_dpcpp_gpu_dpcpp'
#ENV DPL_ROOT='/opt/intel/oneapi/dpl/2022.1.0'
#ENV FI_PROVIDER_PATH='/opt/intel/oneapi/mpi/2021.9.0//libfabric/lib/prov:/usr/lib64/libfabric'
#ENV FPGA_VARS_ARGS=''
#ENV FPGA_VARS_DIR='/opt/intel/oneapi/compiler/2023.1.0/linux/lib/oclfpga'
#ENV GDB_INFO='/opt/intel/oneapi/debugger/2023.1.0/documentation/info/'
#ENV INFOPATH='/opt/intel/oneapi/debugger/2023.1.0/gdb/intel64/lib'
#ENV INTELFPGAOCLSDKROOT='/opt/intel/oneapi/compiler/2023.1.0/linux/lib/oclfpga'
#ENV INTEL_PYTHONHOME='/opt/intel/oneapi/debugger/2023.1.0/dep'
#ENV IPPCP_TARGET_ARCH='intel64'
#ENV IPPCRYPTOROOT='/opt/intel/oneapi/ippcp/2021.7.0'
#ENV IPPROOT='/opt/intel/oneapi/ipp/2021.8.0'
#ENV IPP_TARGET_ARCH='intel64'
#ENV I_MPI_ROOT='/opt/intel/oneapi/mpi/2021.9.0'
#ENV LD_LIBRARY_PATH='/opt/intel/oneapi/tbb/2021.9.0/env/../lib/intel64/gcc4.8:/opt/intel/oneapi/mpi/2021.9.0//libfabric/lib:/opt/intel/oneapi/mpi/2021.9.0//lib/release:/opt/intel/oneapi/mpi/2021.9.0//lib:/opt/intel/oneapi/mkl/2023.1.0/lib/intel64:/opt/intel/oneapi/ippcp/2021.7.0/lib/intel64:/opt/intel/oneapi/ipp/2021.8.0/lib/intel64:/opt/intel/oneapi/dnnl/2023.1.0/cpu_dpcpp_gpu_dpcpp/lib:/opt/intel/oneapi/debugger/2023.1.0/gdb/intel64/lib:/opt/intel/oneapi/debugger/2023.1.0/libipt/intel64/lib:/opt/intel/oneapi/debugger/2023.1.0/dep/lib:/opt/intel/oneapi/dal/2023.1.0/lib/intel64:/opt/intel/oneapi/compiler/2023.1.0/linux/lib:/opt/intel/oneapi/compiler/2023.1.0/linux/lib/x64:/opt/intel/oneapi/compiler/2023.1.0/linux/lib/oclfpga/host/linux64/lib:/opt/intel/oneapi/compiler/2023.1.0/linux/compiler/lib/intel64_lin:/opt/intel/oneapi/ccl/2021.9.0/lib/cpu_gpu_dpcpp:$LD_LIBRARY_PATH'
#ENV LIBRARY_PATH='/opt/intel/oneapi/tbb/2021.9.0/env/../lib/intel64/gcc4.8:/opt/intel/oneapi/mpi/2021.9.0//libfabric/lib:/opt/intel/oneapi/mpi/2021.9.0//lib/release:/opt/intel/oneapi/mpi/2021.9.0//lib:/opt/intel/oneapi/mkl/2023.1.0/lib/intel64:/opt/intel/oneapi/ippcp/2021.7.0/lib/intel64:/opt/intel/oneapi/ipp/2021.8.0/lib/intel64:/opt/intel/oneapi/dnnl/2023.1.0/cpu_dpcpp_gpu_dpcpp/lib:/opt/intel/oneapi/dal/2023.1.0/lib/intel64:/opt/intel/oneapi/compiler/2023.1.0/linux/compiler/lib/intel64_lin:/opt/intel/oneapi/compiler/2023.1.0/linux/lib:/opt/intel/oneapi/ccl/2021.9.0/lib/cpu_gpu_dpcpp'
#ENV MANPATH='/opt/intel/oneapi/mpi/2021.9.0/man:/opt/intel/oneapi/debugger/2023.1.0/documentation/man:/opt/intel/oneapi/compiler/2023.1.0/documentation/en/man/common::'
#ENV MKLROOT='/opt/intel/oneapi/mkl/2023.1.0'
#ENV NLSPATH='/opt/intel/oneapi/mkl/2023.1.0/lib/intel64/locale/%l_%t/%N:/opt/intel/oneapi/compiler/2023.1.0/linux/compiler/lib/intel64_lin/locale/%l_%t/%N'
#ENV OCL_ICD_FILENAMES='libintelocl_emu.so:libalteracl.so:/opt/intel/oneapi/compiler/2023.1.0/linux/lib/x64/libintelocl.so'
#ENV ONEAPI_ROOT='/opt/intel/oneapi'
#ENV PATH='/opt/intel/oneapi/vtune/2023.1.0/bin64:/opt/intel/oneapi/mpi/2021.9.0//libfabric/bin:/opt/intel/oneapi/mpi/2021.9.0//bin:/opt/intel/oneapi/mkl/2023.1.0/bin/intel64:/opt/intel/oneapi/dev-utilities/2021.9.0/bin:/opt/intel/oneapi/debugger/2023.1.0/gdb/intel64/bin:/opt/intel/oneapi/compiler/2023.1.0/linux/lib/oclfpga/bin:/opt/intel/oneapi/compiler/2023.1.0/linux/bin/intel64:/opt/intel/oneapi/compiler/2023.1.0/linux/bin:/opt/intel/oneapi/advisor/2023.1.0/bin64:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH'
#ENV PKG_CONFIG_PATH='/opt/intel/oneapi/vtune/2023.1.0/include/pkgconfig/lib64:/opt/intel/oneapi/tbb/2021.9.0/env/../lib/pkgconfig:/opt/intel/oneapi/mpi/2021.9.0/lib/pkgconfig:/opt/intel/oneapi/mkl/2023.1.0/lib/pkgconfig:/opt/intel/oneapi/ippcp/2021.7.0/lib/pkgconfig:/opt/intel/oneapi/dpl/2022.1.0/lib/pkgconfig:/opt/intel/oneapi/dnnl/2023.1.0/cpu_dpcpp_gpu_dpcpp/../lib/pkgconfig:/opt/intel/oneapi/dal/2023.1.0/lib/pkgconfig:/opt/intel/oneapi/compiler/2023.1.0/lib/pkgconfig:/opt/intel/oneapi/ccl/2021.9.0/lib/pkgconfig:/opt/intel/oneapi/advisor/2023.1.0/include/pkgconfig/lib64:'
#ENV PYTHONPATH='/opt/intel/oneapi/advisor/2023.1.0/pythonapi'
#ENV SETVARS_COMPLETED='1'
#ENV TBBROOT='/opt/intel/oneapi/tbb/2021.9.0/env/..'
#ENV VTUNE_PROFILER_2023_DIR='/opt/intel/oneapi/vtune/2023.1.0'
#ENV VTUNE_PROFILER_DIR='/opt/intel/oneapi/vtune/2023.1.0'
#
