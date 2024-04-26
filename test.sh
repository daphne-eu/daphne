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

# This script builds and runs the test cases for the DAPHNE prototype.
#
# Arguments to this script are forwarded to the test-case executable, i.e., all
# arguments to the catch2 main-function can be used. Invoke `./test.sh -h` or
# see https://github.com/catchorg/Catch2/blob/v2.x/docs/command-line.md for a
# reference.
#
# Invoke this script in the root directory "prototype".

# Stop immediately if any command fails.
set -e

catch2_options=""
BUILD_CUDA=""
BUILD_FPGAOPENCL=""
BUILD_MPI=""
BUILD_DEBUG=""
BUILD_DAPHNE=1

while [[ $# -gt 0 ]]; do
    key=$1
    shift
    case $key in
        --cuda)
            echo using CUDA
            export BUILD_CUDA="--cuda"
            ;;
        --fpgaopencl)
            echo using FPGAOPENCL
            export BUILD_FPGAOPENCL="--fpgaopencl"
            ;;
        --mpi)
            echo using MPI
            export BUILD_MPI="--mpi"
            ;;
        --debug)
            echo building DEBUG version
            export BUILD_DEBUG="--debug"
            ;;
        -nb | --no-build)
            BUILD_DAPHNE=0
            ;;
        *)
            catch2_options="${catch2_options} ${key}"
            ;;
    esac
done

# Build tests.
if [ $BUILD_DAPHNE -gt 0 ]; then
  ./build.sh $BUILD_CUDA $BUILD_FPGAOPENCL $BUILD_MPI $BUILD_DEBUG --target run_tests
fi

# Preparations for running DaphneLib (Python API) tests and MLIR codegen tests (LLVM LIT)
export PYTHONPATH="$PYTHONPATH:$PWD/src/api/python/:/usr/lib/llvm-10/build/utils/lit/"
export DAPHNELIB_DIR_PATH=$PWD/lib
export PATH=$PWD/bin:/usr/lib/llvm-10/bin:$PATH
export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH

# Silence TensorFlow warnings in DaphneLib test cases.
export TF_CPP_MIN_LOG_LEVEL=3

# this speeds up the vectorized tests
export OPENBLAS_NUM_THREADS=1

# Find out if TensorFlow and PyTorch are available in Python and set environment
# variables accordingly. Used to switch the DaphneLib test cases for data transfer
# with these libraries on/off.
# As an alternative to environment variables, we could use a custom catch2 main
# function with custom CLI arguments, but that seems to be more complicated.
if python3 -c "import tensorflow" 2> /dev/null; then
    export DAPHNE_DEP_AVAIL_TENSFORFLOW=1
else
    export DAPHNE_DEP_AVAIL_TENSFORFLOW=0
fi
if python3 -c "import torch" 2> /dev/null; then
    export DAPHNE_DEP_AVAIL_PYTORCH=1
else
    export DAPHNE_DEP_AVAIL_PYTORCH=0
fi

# Run tests.
# shellcheck disable=SC2086
./bin/run_tests $catch2_options

set +e