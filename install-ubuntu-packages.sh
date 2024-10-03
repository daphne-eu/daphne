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
UbuntuVersion=$(lsb_release -sr)

# Ubuntu Version is not 18.04, 20.04, 22.04 or 24.04
function promptContinueOnUnknownVersion {
    read -p "Support for Ubuntu $UbuntuVersion is not guaranteed - attempt to proceed anyway? (y/n) " res
    case "$res" in
        "y"|"Y")
            sudo apt install cmake llvm-15-tools
            ;;
        *)
            exit 1
            ;;
    esac
}

# Handle version dependent packages
case "$UbuntuVersion" in
    "18.04")
        echo "Ubuntu $UbuntuVersion requires a custom toolchain. Please refer to the documentation (https://github.com/daphne-eu/daphne/blob/main/doc/GettingStarted.md)."
        exit 1
        ;;
    "20.04")
        sudo snap install cmake --classic
        sudo apt install llvm-10-tools
        ;;
    "22.04")
        sudo apt install cmake llvm-15-tools
        ;;
    "24.04")
        sudo apt install cmake llvm-18-tools
        ;;
    *)
        promptContinueOnUnknownVersion
        ;;
esac

# Install version independent packages
sudo apt install build-essential clang default-jdk-headless gfortran git jq \
 libpfm4-dev libssl-dev lld ninja-build pkg-config python3-numpy python3-pandas \
 unzip uuid-dev wget zlib1g-dev