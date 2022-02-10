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
        OUTPUT ${PROJECT_BINARY_DIR}/src/runtime/distributed/coordinator/kernels/kernels.cpp
        COMMAND python3 ARGS genKernelInst.py distributedKernels.json ${PROJECT_BINARY_DIR}/src/runtime/distributed/coordinator/kernels/kernels.cpp
        MAIN_DEPENDENCY ${PROJECT_SOURCE_DIR}/src/runtime/distributed/coordinator/kernels/distributedKernels.json
        DEPENDS ${PROJECT_SOURCE_DIR}/src/runtime/distributed/coordinator/kernels/genKernelInst.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/runtime/distributed/coordinator/kernels/
)

# The library of pre-compiled kernels. Will be linked into the JIT-compiled
# user program.
add_library(DistributedKernels SHARED
            ${PROJECT_BINARY_DIR}/src/runtime/distributed/coordinator/kernels/kernels.cpp
            )
target_link_libraries(DistributedKernels PRIVATE DataStructures IO Proto ProtoDataConverter LLVMSupport) 
# ${OPENBLAS_LIBRARIES})
#set_target_properties(AllKernels PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
