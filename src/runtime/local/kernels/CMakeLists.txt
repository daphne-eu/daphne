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
add_custom_command(
        OUTPUT ${PROJECT_BINARY_DIR}/src/runtime/local/kernels/kernels.cpp
        COMMAND python3 ARGS genKernelInst.py kernels.json ${PROJECT_BINARY_DIR}/src/runtime/local/kernels/kernels.cpp
        MAIN_DEPENDENCY ${PROJECT_SOURCE_DIR}/src/runtime/local/kernels/kernels.json
        DEPENDS ${PROJECT_SOURCE_DIR}/src/runtime/local/kernels/genKernelInst.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/runtime/local/kernels/
)

# The library of pre-compiled kernels. Will be linked into the JIT-compiled
# user program.
add_library(AllKernels SHARED
            ${PROJECT_BINARY_DIR}/src/runtime/local/kernels/kernels.cpp
            Pooling.cpp)
target_link_libraries(AllKernels PUBLIC DataStructures IO Proto ProtoDataConverter LLVMSupport
        ${OPENBLAS_LIBRARIES})
#set_target_properties(AllKernels PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

# The library of pre-compiled CUDA kernels
if(USE_CUDA AND CMAKE_CUDA_COMPILER)
    target_include_directories(AllKernels PUBLIC ${CUDAToolkit_INCLUDE_DIRS})

    set(PREFIX ${PROJECT_SOURCE_DIR}/src/runtime/local/)
    set(CUDAKernels_SRC
            ${PREFIX}/context/CUDAContext.cpp
            ${PREFIX}/kernels/CUDA_Activation.cpp
            ${PREFIX}/kernels/CUDA_Affine.cpp
            ${PREFIX}/kernels/CUDA_BatchNorm.cpp
            ${PREFIX}/kernels/CUDA_BiasAdd.cpp
            ${PREFIX}/kernels/CUDA_Convolution.cpp
            ${PREFIX}/kernels/CUDA_kernel_instantiations.cpp
            ${PREFIX}/kernels/CUDA_MatMul.cpp
            ${PREFIX}/kernels/CUDA_Pooling.cpp
            ${PREFIX}/kernels/CUDA_Softmax.cpp)

    add_library(CUDAKernels SHARED ${CUDAKernels_SRC})

    # search "custom" cudnn lib in CUDA SDK dir
    set(lib_name cudnn)
    find_library(CUDA_${lib_name}_LIBRARY NAMES ${lib_name} HINTS ${CUDAToolkit_LIBRARY_DIR} ENV CUDA_PATH
            PATH_SUFFIXES nvidia/current lib64 lib/x64 lib)

    target_link_libraries(CUDAKernels PRIVATE DataStructures CUDA::cudart CUDA::cublasLt CUDA::cublas CUDA::cusparse
            ${CUDA_cudnn_LIBRARY})
    #    set_target_properties(CUDAKernels PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
endif()