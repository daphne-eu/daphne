# Copyright 2023 The DAPHNE Consortium
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

# ************************************************************************
# Set required libraries
# ************************************************************************

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRDaphne
        DaphneDSLParser
        DaphneIrExecutor
        DaphneConfigParser
        DaphneCatalogParser
        DaphneMetaDataParser
        Util
        WorkerImpl
)

if(USE_FPGAOPENCL)
    list(APPEND LIBS FPGAOPENCLKernels)
endif()

# ************************************************************************
# DaphneInternal static library
# ************************************************************************
# A static library used by both the command line executable daphne and the
# shared library daphnelib.

add_library(DaphneInternal STATIC daphne_internal.cpp) # DaphneUserConfig.h
set_target_properties(DaphneInternal PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
target_link_libraries(DaphneInternal PRIVATE ${LIBS} ${MPI_LIBRARIES})

if(USE_CUDA AND CMAKE_CUDA_COMPILER)
    #    list(APPEND LIBS CUDAKernels)
    target_include_directories(DaphneInternal PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
endif()