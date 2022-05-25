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

    # Delete entire directories.
    local dirs=("build" \
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

    # Delete individual files.
    if [ -f $llvmCommitFilePath ]
    then
        echo "---- cleanup $llvmCommitFilePath"
        rm -f $llvmCommitFilePath
    else
        echo "---- cleanup $llvmCommitFilePath - non-existing"
    fi
}

#******************************************************************************
# Set some paths
#******************************************************************************

initPwd=$(pwd)
thirdpartyPath=$initPwd/thirdparty
llvmCommitFilePath=$thirdpartyPath/llvm-last-built-commit.txt

# a hotfix, to solve issue #216 @todo investigate possible side effects
installLibDir=lib

#******************************************************************************
# Parse arguments
#******************************************************************************

# Defaults.
target="daphne"

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
# But only if this is a git repo.
if [ -d .git ]
then
    git submodule update --init --recursive
fi


#******************************************************************************
# Handle third-party material
#******************************************************************************

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
    # Github disabled the unauthenticated git:// protocol, patch antlr4 to use https://
    # until we upgrade to antlr4-4.9.3+
    sed -i 's#git://github.com#https://github.com#' runtime/CMakeLists.txt
    rm -rf ./build
    mkdir -p build
    mkdir -p run
    cd build
    cmake .. -G Ninja  -DANTLR_JAR_LOCATION=../$antlrJarName -DANTLR4_INSTALL=ON -DCMAKE_INSTALL_PREFIX=../run/usr/local -DCMAKE_INSTALL_LIBDIR=$installLibDir
    cmake --build . --target install
fi
cd $pwdBeforeAntlr

# catch2 (unit test framework).
# Download catch2 release zip (if necessary), and unpack the single header file
# (if necessary).
catch2Name=catch2
catch2Version=2.13.8 # for upgrades, it suffices to simply change the version here
catch2ZipName=v$catch2Version.zip
catch2SingleHeaderName=catch.hpp
mkdir --parents $catch2Name
cd $catch2Name
if [ ! -f $catch2ZipName ] || [ ! -f $catch2SingleHeaderName ]
then
    wget https://github.com/catchorg/Catch2/archive/refs/tags/$catch2ZipName
    unzip -p $catch2ZipName "Catch2-$catch2Version/single_include/catch2/catch.hpp" \
        > $catch2SingleHeaderName
fi
cd ..

# OpenBLAS (basic linear algebra subprograms)
pwdBeforeOpenBlas=$(pwd)
openBlasDirName=OpenBLAS
openBlasVersion=0.3.19
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
    make -j
    make install PREFIX=../$openBlasInstDirName
fi
cd $pwdBeforeOpenBlas

# nlohmann/json (library for JSON parsing)
pwdBeforeNlohmannjson=$(pwd)
nlohmannjsonDirName=nlohmannjson
nlohmannjsonVersion=3.10.5
nlohmannjsonSingleHeaderName=json.hpp
mkdir --parents $nlohmannjsonDirName
cd $nlohmannjsonDirName
if [ ! -f $nlohmannjsonSingleHeaderName ]
then
    wget https://github.com/nlohmann/json/releases/download/v$nlohmannjsonVersion/$nlohmannjsonSingleHeaderName
fi
cd $pwdBeforeNlohmannjson

# gRPC
grpcDirName=grpc
grpcInstDir=$(pwd)/$grpcDirName/installed
if [ ! -d $grpcDirName ]
then
    # Download gRPC source code.
    git clone --recurse-submodules -b v1.38.0 https://github.com/grpc/grpc $grpcDirName
    mkdir -p "$grpcInstDir"

    pushd $grpcDirName

    # Install gRPC and its dependencies.
    mkdir -p "cmake/build"
    pushd "cmake/build"
    cmake \
      -DCMAKE_INSTALL_PREFIX="$grpcInstDir" \
      -DCMAKE_BUILD_TYPE=Release \
      -DgRPC_INSTALL=ON \
      -DgRPC_BUILD_TESTS=OFF \
      ../..
      #-DgRPC_ABSL_PROVIDER=package \
    make -j4 install
    popd

    # Install abseil.
    mkdir -p third_party/abseil-cpp/cmake/build
    pushd third_party/abseil-cpp/cmake/build
    cmake -DCMAKE_INSTALL_PREFIX="$grpcInstDir" \
        -DCMAKE_POSITION_INDEPENDENT_CODE=TRUE \
        ../..
    make -j
    make install
    popd

    popd
fi

#------------------------------------------------------------------------------
# Build MLIR
#------------------------------------------------------------------------------
# We rarely need to build MLIR/LLVM, only during the first build of the
# prototype and after upgrades of the LLVM sub-module. To avoid unnecessary
# builds (which take several seconds even if there is nothing to do), we store
# the LLVM commit hash we built into a file, and only rebuild MLIR/LLVM if this
# file does not exist (first build of the prototype) or does not contain the
# expected hash (upgrade of the LLVM sub-module).

llvmName=llvm-project
cd $llvmName
llvmCommit="llvmCommit-local-none"
if [ -e .git ] # Note: .git in the submodule is not a directory.
then
    llvmCommit=$(git log -1 --format=%H)
fi

if [ ! -f $llvmCommitFilePath ] || [ $(cat $llvmCommitFilePath) != $llvmCommit ]
then
    echo "Need to build MLIR/LLVM."
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
    echo $llvmCommit > $llvmCommitFilePath
else
    echo "No need to build MLIR/LLVM again."
fi

#------------------------------------------------------------------------------
# Go back
#------------------------------------------------------------------------------

cd $initPwd


# *****************************************************************************
# Build the DAPHNE prototype.
# *****************************************************************************

mkdir --parents build
cd build
cmake -G Ninja ..  -DCMAKE_BUILD_TYPE=Release \
    -DMLIR_DIR=$thirdpartyPath/$llvmName/build/lib/cmake/mlir/ \
    -DLLVM_DIR=$thirdpartyPath/$llvmName/build/lib/cmake/llvm/ \
    -DANTLR4_RUNTIME_DIR=$thirdpartyPath/$antlrDirName/$antlrCppRuntimeDirName \
    -DANTLR4_JAR_LOCATION=$thirdpartyPath/$antlrDirName/$antlrJarName \
    -DOPENBLAS_INST_DIR=$thirdpartyPath/$openBlasDirName/$openBlasInstDirName \
    -DCMAKE_PREFIX_PATH="$grpcInstDir" \
    -DCMAKE_INSTALL_LIBDIR=$installLibDir \
    -DUSE_CUDA=OFF \
    -DUSE_ONEAPI=ON \
    -DUSE_ARROW=OFF

cmake --build . --target $target
cd -

# build OneAPI library in a separate step because of the changed CXX compiler
CXX=dpcpp cmake -S src/runtime/local/kernels/ONEAPI -B build/oneapi-kernels -G Ninja -DCMAKE_PREFIX_PATH=build -DUSE_ONEAPI=ON
cmake --build build/oneapi-kernels

set +e
