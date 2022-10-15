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

#       * more properly set up paths (ugly relative include path below)
#cmake_minimum_required(VERSION 3.21)

#project(FPGAOPENCLkernels)

include_directories($ENV{QUARTUSDIR}/hld/examples_aoc/common/inc/ $ENV{QUARTUSDIR}/hld/host/include/)

set(PREFIX ${PROJECT_SOURCE_DIR}/src/runtime/local/kernels/FPGAOPENCL)
#message(STATUS "cmake fpga test! $ENV{QUARTUSDIR}")
add_custom_command(
    OUTPUT ${PROJECT_BINARY_DIR}/FPGAOPENCLkernels.cpp
    COMMAND python3 ARGS genKernelInst.py kernels.json
    ${PROJECT_BINARY_DIR}/FPGAOPENCLkernels.cpp FPGAOPENCL
    MAIN_DEPENDENCY ${PREFIX}/../kernels.json
    DEPENDS ${PREFIX}/../genKernelInst.py
    WORKING_DIRECTORY ${PREFIX}/..
)

#set(PREFIX ${PROJECT_SOURCE_DIR}/src/runtime/local/kernels/FPGAOPENCL)
set(FPGAOPENCLKernels_SRC
	${PREFIX}/../../context/FPGAContext.cpp
 	${PREFIX}/../../context/FPGAContext.h
	${PREFIX}/CreateFPGAContext.h
        ${PREFIX}/MatMul.h
	${PREFIX}/gemm_interface.cpp
	${PREFIX}/gemm_interface.h
	${PROJECT_BINARY_DIR}/FPGAOPENCLkernels.cpp
)

add_library(FPGAOPENCLKernels SHARED ${FPGAOPENCLKernels_SRC})
set_target_properties(FPGAOPENCLKernels PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

target_include_directories(FPGAOPENCLKernels PUBLIC ${PROJECT_SOURCE_DIR}/src/)

target_link_libraries(FPGAOPENCLKernels PUBLIC AllKernels LLVMSupport $ENV{QUARTUSDIR}/hld/linux64/lib/libOpenCL.so)

