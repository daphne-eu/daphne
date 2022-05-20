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

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRDaphne
        DaphneDSLParser
        DaphneIrExecutor
        DaphneConfigParser
        DaphneMetaDataParser
)

# These dependencies are linked at runtime
set(LIB_DEPS MLIRDaphneOpsIncGen AllKernels)

if(USE_CUDA AND CMAKE_CUDA_COMPILER)
    list(APPEND LIBS CUDA::cudart)
    list(APPEND LIB_DEPS CUDAKernels)
endif()

add_llvm_executable(daphne daphne.cpp DaphneUserConfig.h DEPENDS ${LIB_DEPS})

llvm_update_compile_flags(daphne)
target_link_libraries(daphne PRIVATE ${LIBS})

mlir_check_all_link_libraries(daphne)
