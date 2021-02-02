#!/bin/bash

set -e

#******************************************************************************
# Download thirdparty material
#******************************************************************************

thirdpartyPath=thirdparty
mkdir --parents $thirdpartyPath
oldPwd=$(pwd)
cd $thirdpartyPath

# Download antlr4 jar if it does not exist yet.
antlrName=antlr-4.9.1-complete.jar
if [ ! -f $antlrName ]
then
    wget https://www.antlr.org/download/$antlrName
fi

# Clone and build LLVM/MLIR if it does not exist yet.
llvmName=llvm-project
if [ ! -d $llvmName ]
then
    git clone https://github.com/llvm/$llvmName.git
    mkdir $llvmName/build
    cd $llvmName/build
    cmake -G Ninja ../llvm \
       -DLLVM_ENABLE_PROJECTS=mlir \
       -DLLVM_BUILD_EXAMPLES=ON \
       -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
       -DCMAKE_BUILD_TYPE=Release \
       -DLLVM_ENABLE_ASSERTIONS=ON \
       -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON
    cmake --build . --target check-mlir
fi

cd $oldPwd

# *****************************************************************************
# Build the DAPHNE prototype.
# *****************************************************************************

mkdir --parents build
cd build
cmake -G Ninja .. -DMLIR_DIR=$thirdpartyPath/$llvmName/build/lib/cmake/mlir/
cmake --build . --target daphnec

set +e