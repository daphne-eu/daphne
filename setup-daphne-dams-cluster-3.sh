#!/bin/bash

# *****************************************************************************
# This script builds DAPHNE including all dependencies.
#
# It is intended to be used on a scale-out node of the DAMS Lab cluster.
# It contains a few work-arounds that are currently needed in this environment.
# *****************************************************************************

# Stop if any command fails.
set -e

# Create a Python virtual environment to make numpy and pandas available.
python3 -m venv daphne-venv
source daphne-venv/bin/activate
pip install numpy pandas

# Use gcc-9/g++-9 for building the dependencies (work-around because we don't have gfortran-11 yet).
export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9

# Build the dependencies and DAPHNE (just some random target that does not require C++-20, s.t. we can build it with g++-9).
./build.sh --target DaphneDSLParser

# Remove the DAPHNE build artifacts.
./build.sh --clean -y

# Use gcc-11/g++-11 for building DAPHNE.
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11

# Build DAPHNE (including all test cases).
./build.sh --target run_tests

# Run the test cases.
./test.sh -d yes

set +e

# Each time you log in to your node:
 source daphne-venv/bin/activate
 export CC=/usr/bin/gcc-11
 export CXX=/usr/bin/g++-11
