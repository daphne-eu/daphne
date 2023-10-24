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

set(PREFIX ${PROJECT_BINARY_DIR}/src/runtime/local/kernels/SIMDOperators)

### Specify source files
#set(SIMDOperators_kernels_SRC
#    ${PREFIX}/SIMDOperators_kernels.cpp)

### Add SIMDOperators API as lib
#add_library(SIMDOperatorsKernels SHARED ${SIMDOperators_kernels_SRC})
#set_target_properties(SIMDOperatorsKernels PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
### Link against common libs
target_compile_options(AllKernels PUBLIC "-mfma")
target_link_libraries(AllKernels PUBLIC SIMDOperators)
#target_link_libraries(SIMDOperatorsKernels PUBLIC ${LIBS})

add_executable(microbenchmark_ssb ./MicrobenchmarkSIMDOpsSSBQ11.cpp)
target_link_libraries(microbenchmark_ssb PUBLIC SIMDOperators ${LIBS} )
#target_compile_options(microbenchmark_ssb PUBLIC -g -O0)
set_target_properties(microbenchmark_ssb PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(microbenchmark_proj ./MicrobenchmarkProjectionPathReplacement.cpp)
target_link_libraries(microbenchmark_proj PUBLIC SIMDOperators ${LIBS} )
#target_compile_options(microbenchmark_proj PUBLIC -g -O0)
set_target_properties(microbenchmark_proj PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
