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

include_directories(${PROJECT_SOURCE_DIR}/test) # for "tags.h"
include_directories(${PROJECT_SOURCE_DIR}/thirdparty/catch2) # for "catch.hpp"

add_executable(run_tests
        run_tests.cpp
        
        runtime/local/datastructures/CSRMatrixTest.cpp
        runtime/local/datastructures/DenseMatrixTest.cpp
        runtime/local/datastructures/FrameTest.cpp
        runtime/local/datastructures/MatrixTest.cpp

        runtime/local/io/ReadCsv.cpp

        runtime/local/kernels/AggAllTest.cpp
        runtime/local/kernels/AggColTest.cpp
        runtime/local/kernels/AggRowTest.cpp
        runtime/local/kernels/CastScaTest.cpp
        runtime/local/kernels/CheckEqTest.cpp
        runtime/local/kernels/EwBinaryMatTest.cpp
        runtime/local/kernels/EwBinaryMatScaTest.cpp
        runtime/local/kernels/EwBinaryScaTest.cpp
        runtime/local/kernels/EwUnaryScaTest.cpp
        runtime/local/kernels/MatMulTest.cpp
        runtime/local/kernels/RandMatrixTest.cpp
        runtime/local/kernels/TransposeTest.cpp
	runtime/local/kernels/ReplaceTest.cpp
)
target_link_libraries(run_tests PRIVATE
    DataStructures
    ${OPENBLAS_LIBRARIES}
)
