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
    echo "  --clean           Remove all temporary build directories for a fresh build"
}

#******************************************************************************
# Clean build directories
#******************************************************************************

function cleanBuildDirs {
    echo "-- Cleanup of build directories pwd=$(pwd) ..."
    dirs=("build" \
        "thirdparty/llvm-project/build" \
        "thirdparty/antlr")
    for ((i=0; i<${#dirs[@]}; i++))
    do
        if [ -d ${dirs[$i]} ]
        then
            echo "---- cleanup ${dirs[$i]}"
            rm -rf ${dirs[$i]}
        else
            echo "---- cleanup ${dirs[$i]} - non-existing"
        fi
    done
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
        --clean)
            cleanBuildDirs
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
pwdBeforeAntlr=$(pwd)
antlrDirName=antlr
antlrVersion=4.9.2
antlrJarName=antlr-${antlrVersion}-complete.jar
antlrCppRuntimeDirName=antlr4-cpp-runtime-${antlrVersion}-source
antlrCppRuntimeZipName=$antlrCppRuntimeDirName.zip
mkdir --parents $antlrDirName
cd $antlrDirName
# Download antlr4 jar if it does not exist yet.
if [ ! -f $antlrJarName ]
then
    wget https://www.antlr.org/download/$antlrJarName
fi
# Download and build antlr4 C++ run-time if it does not exist yet.
if [ ! -f $antlrCppRuntimeZipName ]
then
    wget https://www.antlr.org/download/$antlrCppRuntimeZipName
    mkdir --parents $antlrCppRuntimeDirName
    unzip $antlrCppRuntimeZipName -d $antlrCppRuntimeDirName
    cd $antlrCppRuntimeDirName
    mkdir build
    mkdir run
    cd build
    cmake .. -DANTLR_JAR_LOCATION=../$antlrJarName -DANTLR4_INSTALL=ON
    make
    DESTDIR=../run make install
fi
cd $pwdBeforeAntlr

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

# OpenBLAS (basic linear algebra subprograms)
pwdBeforeOpenBlas=$(pwd)
openBlasDirName=OpenBLAS
openBlasVersion=0.3.15
openBlasZipName=OpenBLAS-$openBlasVersion.zip
openBlasInstDirName=installed
mkdir --parents $openBlasDirName
cd $openBlasDirName
if [ ! -f $openBlasZipName ]
then
    wget https://github.com/xianyi/OpenBLAS/releases/download/v$openBlasVersion/$openBlasZipName
    unzip $openBlasZipName
    mkdir --parents $openBlasInstDirName
    cd OpenBLAS-$openBlasVersion
    make
    make install PREFIX=../$openBlasInstDirName
fi
cd $pwdBeforeOpenBlas

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
   -DLLVM_BUILD_EXAMPLES=OFF \
   -DLLVM_TARGETS_TO_BUILD="X86" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON \
   -DLLVM_ENABLE_RTTI=ON
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
cmake -G Ninja .. \
    -DMLIR_DIR=$thirdpartyPath/$llvmName/build/lib/cmake/mlir/ \
    -DLLVM_DIR=$thirdpartyPath/$llvmName/build/lib/cmake/llvm/ \
    -DANTLR4_RUNTIME_DIR=$(pwd)/../$thirdpartyPath/$antlrDirName/$antlrCppRuntimeDirName \
    -DANTLR4_JAR_LOCATION=$(pwd)/../$thirdpartyPath/$antlrDirName/$antlrJarName \
    -DOPENBLAS_INST_DIR=$(pwd)/../$thirdpartyPath/$openBlasDirName/$openBlasInstDirName
cmake --build . --target $target


set +e