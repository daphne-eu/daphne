#!/bin/bash

# This script builds the DAPHNE prototype.
#
# It also downloads and builds all required third-party material. Simply invoke
# it without any arguments.


# Stop immediately if any command fails.
set -e


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