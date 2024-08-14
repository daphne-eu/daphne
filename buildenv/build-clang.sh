#!/usr/bin/env bash

source ./env.sh
ROOTDIR=$(pwd)

if [ ! -f $DLC/llvmorg-${LLVM_VERSION}.tar.gz ]; then
      wget https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-${LLVM_VERSION}.tar.gz -P $DLC
fi 

tar xf $DLC/llvmorg-${LLVM_VERSION}.tar.gz
cd llvm-project-llvmorg-${LLVM_VERSION}
mkdir --parents build-clang
cd build-clang
cmake -G Ninja ../llvm \
     -DLLVM_ENABLE_PROJECTS=clang \
     -DLLVM_BUILD_EXAMPLES=OFF \
     -DLLVM_TARGETS_TO_BUILD="X86" \
     -DCMAKE_BUILD_TYPE=Release \
     -DLLVM_ENABLE_ASSERTIONS=ON \
     -DLLVM_ENABLE_RTTI=ON \
     -DCMAKE_INSTALL_PREFIX=$ROOTDIR/installed
cmake --build . --target clang
cmake --build . --target install/strip
cd ..
mkdir --parents build-lld
cd build-lld
cmake -G Ninja ../llvm \
      -DLLVM_ENABLE_PROJECTS=lld \
      -DLLVM_BUILD_EXAMPLES=OFF \
      -DLLVM_TARGETS_TO_BUILD="X86" \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DLLVM_ENABLE_RTTI=ON \
      -DCMAKE_INSTALL_PREFIX=$ROOTDIR/installed
cmake --build . --target lld
cmake --build . --target install/strip

cd $ROOTDIR
