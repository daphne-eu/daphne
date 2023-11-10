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

# This is a convenience script to install the required packages on Ubuntu 20+ systems to compile DAPHNE
# On Ubuntu 22.04 you may need to change the version of llvm-10-tools to a newer one, such as llvm-15-tools.
sudo apt install build-essential clang cmake git libssl-dev libpfm4-dev lld ninja-build pkg-config python3-numpy \
 python3-pandas default-jdk-headless gfortran uuid-dev wget unzip jq llvm-10-tools
