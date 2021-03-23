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

# This script builds the DAPHNE prototype.
#
# It also downloads and builds all required third-party material. Simply invoke
# it without any arguments.


# Stop immediately if any command fails.
set -e

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

# Download antlr4 jar if it does not exist yet.
antlrName=antlr-4.9.1-complete.jar
if [ ! -f $antlrName ]
then
    wget https://www.antlr.org/download/$antlrName
fi

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
cmake --build . --target daphnec


set +e