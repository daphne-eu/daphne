#!/bin/bash

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

# Stop immediately if any command fails.
set -e

#******************************************************************************
# Help message
#******************************************************************************

function printHelp {
    echo "Build the DAPHNE prototype."
    echo ""
    echo "Usage: $0 [-h|--help] [--target TARGET]"
    echo ""
    echo "This includes downloading and building all required third-party "
    echo "material, if necessary. Simply invoke this script without any "
    echo "arguments to build the prototype. You should only invoke it from "
    echo "the prototype's root directory (where this script resides)."
    echo ""
    echo "Optional arguments:"
    echo "  -h, --help        Print this help message and exit."
    echo "  --target TARGET   Build the cmake target TARGET (defaults to '$target')"
}

#******************************************************************************
# Parse arguments
#******************************************************************************

# Defaults.
target="daphnec"

while [[ $# -gt 0 ]]
do
    key=$1
    shift
    case $key in
        -h|--help)
            printHelp
            exit 0
            ;;
        --target)
            target=$1
            shift
            ;;
        *)
            printf "Unknown option: '%s'\n\n" $key
            printHelp
            exit 1
            ;;
    esac
done;


# Make sure that the submodule(s) have been updated since the last clone/pull.
git submodule update --init --recursive


#******************************************************************************
# Handle third-party material
#******************************************************************************

thirdpartyPath=thirdparty
oldPwd=$(pwd)
cd $thirdpartyPath

#------------------------------------------------------------------------------
# Download third-party material if necessary
#------------------------------------------------------------------------------

# antlr4 (parser).
# Download antlr4 jar if it does not exist yet.
antlrName=antlr-4.9.1-complete.jar
if [ ! -f $antlrName ]
then
    wget https://www.antlr.org/download/$antlrName
fi

# catch2 (unit test framework).
# Download catch2 release zip (if necessary), and unpack the single header file
# (if necessary).
catch2Name=catch2
catch2ZipName=v2.13.4.zip
catch2SingleHeaderName=catch.hpp
mkdir --parents $catch2Name
cd $catch2Name
if [ ! -f $catch2ZipName ]
then
    wget https://github.com/catchorg/Catch2/archive/refs/tags/$catch2ZipName
fi
if [ ! -f $catch2SingleHeaderName ]
then
    unzip -p $catch2ZipName "Catch2-2.13.4/single_include/catch2/catch.hpp" \
        > $catch2SingleHeaderName
fi
cd ..

#------------------------------------------------------------------------------
# Build MLIR
#------------------------------------------------------------------------------
# TODO Do this only when it is necessary, since cmake takes some time even if
# there is nothing to do.

llvmName=llvm-project
cd $llvmName
mkdir --parents build
cd build
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON
cmake --build . --target check-mlir

#------------------------------------------------------------------------------
# Go back
#------------------------------------------------------------------------------

cd $oldPwd


# *****************************************************************************
# Build the DAPHNE prototype.
# *****************************************************************************

mkdir --parents build
cd build
cmake -G Ninja .. -DMLIR_DIR=$thirdpartyPath/$llvmName/build/lib/cmake/mlir/
cmake --build . --target $target


set +e