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

set(SOURCES DaphneIrExecutor.cpp DaphneIrExecutor.h)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        ${extension_libs}
        MLIRDaphne
        MLIRDaphneExplain
        MLIRDaphneInference
        MLIRDaphneTransforms
        MLIRExecutionEngine
        MLIRReconcileUnrealizedCasts
        )

add_llvm_library(DaphneIrExecutor ${SOURCES}
        DEPENDS
        MLIRDaphneOpsIncGen
        # The kernels are linked at runtime.
        AllKernels
        )

llvm_update_compile_flags(DaphneIrExecutor)
target_link_libraries(DaphneIrExecutor PUBLIC ${LIBS})
mlir_check_all_link_libraries(DaphneIrExecutor)
