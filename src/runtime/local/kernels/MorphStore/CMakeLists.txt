# Copyright 2022 The DAPHNE Consortium
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

if(USE_MORPHSTORE)
    set(PREFIX ${PROJECT_BINARY_DIR}/src/runtime/local/kernels/MorphStore)

    ### Create Kernel calls using MorphStore API
    add_custom_command(
                OUTPUT ${PREFIX}/MorphStore_kernels.cpp
                COMMAND python3 ARGS genKernelInst.py kernels.json
                        ${PREFIX}/MorphStore_kernels.cpp MorphStore
                MAIN_DEPENDENCY ${PROJECT_SOURCE_DIR}/src/runtime/local/kernels/kernels.json
                DEPENDS ${PROJECT_SOURCE_DIR}/src/runtime/local/kernels/genKernelInst.py
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/runtime/local/kernels/
        )

    ### Specify source files
    set(MorphStore_kernels_SRC
        ${PREFIX}/MorphStore_kernels.cpp)

    ### Add MorphStore API as lib
    add_library(MorphStoreKernels SHARED ${MorphStore_kernels_SRC})
    set_target_properties(MorphStoreKernels PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
    ### Link against common libs
    target_link_libraries(MorphStoreKernels PUBLIC ${LIBS})
    target_add_morphstore(MorphStoreKernels)

endif()
