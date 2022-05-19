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

# Specifies how to generate the file "kernels.cpp" (which resides in the build
# directory) as the basis for the pre-compiled kernels library.

# ToDo: * check required cmake version for OneAPI
#       * more properly set up paths (ugly relative include path below)
cmake_minimum_required(VERSION 3.21)

project(ONEAPIkernels)

add_custom_command(
    OUTPUT ${PROJECT_BINARY_DIR}/ONEAPIkernels.cpp
    COMMAND python3 ARGS genKernelInst.py kernels.json
        ${PROJECT_BINARY_DIR}/ONEAPIkernels.cpp ONEAPI
    MAIN_DEPENDENCY ${PROJECT_SOURCE_DIR}/../kernels.json
    DEPENDS ${PROJECT_SOURCE_DIR}/../genKernelInst.py
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/..
)

set(PREFIX ${PROJECT_SOURCE_DIR}/)
set(ONEAPIKernels_SRC
        ${PREFIX}/MatMul.cpp
        ${PROJECT_BINARY_DIR}/ONEAPIkernels.cpp
)

add_library(ONEAPIKernels SHARED ${ONEAPIKernels_SRC})

target_include_directories(ONEAPIKernels PUBLIC ${PROJECT_SOURCE_DIR}/../../../../)

