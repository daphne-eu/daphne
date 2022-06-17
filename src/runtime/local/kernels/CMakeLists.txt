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


# *****************************************************************************
# DistributedCommunication library
# *****************************************************************************

set(SOURCES DistributedCaller.h)
set(LIBS Proto)

add_library(DistributedCaller ${SOURCES})
target_link_libraries(DistributedCaller PRIVATE ${LIBS})


# The library of pre-compiled CUDA kernels
if(USE_CUDA AND CMAKE_CUDA_COMPILER)
    add_custom_command(
            OUTPUT ${PROJECT_BINARY_DIR}/src/runtime/local/kernels/CUDAkernels.cpp
            COMMAND python3 ARGS genKernelInst.py kernels.json
                    ${PROJECT_BINARY_DIR}/src/runtime/local/kernels/CUDAkernels.cpp CUDA
            MAIN_DEPENDENCY ${PROJECT_SOURCE_DIR}/src/runtime/local/kernels/kernels.json
            DEPENDS ${PROJECT_SOURCE_DIR}/src/runtime/local/kernels/genKernelInst.py
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/runtime/local/kernels/
    )

    set(PREFIX ${PROJECT_SOURCE_DIR}/src/runtime/local/kernels/CUDA)
    set(CUDAKernels_SRC
            ${PREFIX}/../../context/CUDAContext.cpp
            ${PREFIX}/AggCol.cu
            ${PREFIX}/Activation.cpp
            ${PREFIX}/Affine.cpp
            ${PREFIX}/BatchNorm.cpp
            ${PREFIX}/BiasAdd.cpp
            ${PREFIX}/Convolution.cpp
            ${PREFIX}/Pooling.cpp
            ${PREFIX}/Softmax.cpp
            ${PREFIX}/ColBind.cu
            ${PREFIX}/EwBinaryMat.cu
            ${PREFIX}/EwBinaryObjSca.cu
            ${PREFIX}/ExtractCol.cu
            ${PREFIX}/Gemv.cpp
            ${PREFIX}/MatMul.cpp
            ${PREFIX}/Solve.cpp
            ${PREFIX}/Syrk.cu
            ${PREFIX}/Transpose.cpp
            ${PROJECT_BINARY_DIR}/src/runtime/local/kernels/CUDAkernels.cpp
            ${PROJECT_SOURCE_DIR}/src/runtime/local/vectorized/TasksCUDA.cpp
        )

    add_library(CUDAKernels SHARED ${CUDAKernels_SRC})

    # search "custom" cudnn lib in CUDA SDK dir
    set(lib_name cudnn)
    find_library(CUDA_${lib_name}_LIBRARY NAMES ${lib_name} HINTS ${CUDAToolkit_LIBRARY_DIR} ENV CUDA_PATH
            PATH_SUFFIXES nvidia/current lib64 lib/x64 lib)

    target_link_libraries(CUDAKernels PUBLIC DataStructures LLVMSupport CUDA::cudart CUDA::cublasLt CUDA::cublas
            CUDA::cusparse ${CUDA_cudnn_LIBRARY} CUDA::cusolver)
    #    set_target_properties(CUDAKernels PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
endif()

add_custom_command(
        OUTPUT ${PROJECT_BINARY_DIR}/src/runtime/local/kernels/kernels.cpp
        COMMAND python3 ARGS genKernelInst.py kernels.json ${PROJECT_BINARY_DIR}/src/runtime/local/kernels/kernels.cpp CPP
        MAIN_DEPENDENCY ${PROJECT_SOURCE_DIR}/src/runtime/local/kernels/kernels.json
        DEPENDS ${PROJECT_SOURCE_DIR}/src/runtime/local/kernels/genKernelInst.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/runtime/local/kernels/
)

list(APPEND LIBS DataStructures IO ProtoDataConverter BLAS::BLAS)

# The library of pre-compiled kernels. Will be linked into the JIT-compiled user program.
add_library(AllKernels SHARED
        ${PROJECT_BINARY_DIR}/src/runtime/local/kernels/kernels.cpp
        ${PROJECT_SOURCE_DIR}/src/runtime/local/vectorized/Tasks.cpp
        ${PROJECT_SOURCE_DIR}/src/runtime/local/vectorized/MTWrapper_dense.cpp
        ${PROJECT_SOURCE_DIR}/src/runtime/local/vectorized/MTWrapper_sparse.cpp
        Pooling.cpp)
#set_target_properties(AllKernels PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

if(USE_CUDA AND CMAKE_CUDA_COMPILER)
target_include_directories(AllKernels PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
    list(APPEND LIBS CUDAKernels)
else()
    # This can only appear once, so it's either included in CUDAKernels or appended here to AllKernels directly
    list(APPEND LIBS LLVMSupport)
endif()

target_link_libraries(AllKernels PUBLIC ${LIBS})
