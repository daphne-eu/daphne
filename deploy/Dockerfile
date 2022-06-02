# Copyright 2021 The DAPHNE Consortium
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

FROM ubuntu:latest
RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get -y install wget
RUN apt-get -y install cmake
RUN apt-get -y install build-essential libssl-dev
RUN apt-get install -y git
RUN apt-get install -y llvm-10 llvm-10-dev clang-10
RUN apt-get install -y ninja-build
RUN apt-get install -y vim
RUN apt-get install -y nano
RUN mv /usr/bin/clang-10 /usr/bin/clang
RUN mv /usr/bin/clang++-10 /usr/bin/clang++
RUN apt-get install python3.8
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y openjdk-11-jdk
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y pkg-config
RUN apt-get install -y uuid-dev
RUN apt-get install -y lldb-10 libllvm10 llvm-10-runtime
RUN apt-get install -y lld
RUN apt-get install -y libncurses5-dev libncursesw5-dev
RUN mkdir sources && cd sources && wget https://github.com/Kitware/CMake/releases/download/v3.20.2/cmake-3.20.2.tar.gz
RUN cd /sources/ && gunzip cmake-3.20.2.tar.gz && tar -xvf cmake-3.20.2.tar &&  cd cmake-3.20.2
RUN cd /sources/cmake-3.20.2/ && ./bootstrap && make && make install
RUN apt-get install -y unzip
RUN apt-get -y update
RUN apt-get install -y  gfortran
CMD echo "Done"
