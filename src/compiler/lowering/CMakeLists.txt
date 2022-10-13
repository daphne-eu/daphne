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

add_mlir_dialect_library(MLIRDaphneTransforms
    RewriteSqlOpPass.cpp
    DistributeComputationsPass.cpp
    DistributePipelinesPass.cpp
    MarkCUDAOpsPass.cpp
    MarkFPGAOPENCLOpsPass.cpp
    InsertDaphneContextPass.cpp
    ManageObjRefsPass.cpp
    LowerToLLVMPass.cpp
    RewriteToCallKernelOpPass.cpp
    SpecializeGenericFunctionsPass.cpp
    VectorizeComputationsPass.cpp
    WhileLoopInvariantCodeMotionPass.cpp

    DEPENDS
    MLIRDaphneOpsIncGen
    MLIRDaphneTransformsIncGen

    LINK_COMPONENTS
    Core
)
target_link_libraries(MLIRDaphneTransforms PUBLIC
    MLIRStandardToLLVM
    SQLParser
)

if(USE_CUDA AND CMAKE_CUDA_COMPILER)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
endif()
