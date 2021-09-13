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

include_directories(${PROJECT_SOURCE_DIR}/test)
include_directories(${PROJECT_SOURCE_DIR}/thirdparty/catch2) # for "catch.hpp"

add_executable(run_tests
        run_tests.cpp

        api/cli/controlflow/ControlFlowTest.cpp
        api/cli/distributed/DistributedTest.cpp
        api/cli/literals/LiteralsTest.cpp
        api/cli/operations/OperationsTest.cpp
        api/cli/parser/ParserTest.cpp
        api/cli/scoping/ScopingTest.cpp
        api/cli/scriptargs/ScriptArgsTest.cpp
        api/cli/vectorized/MultiThreadedOpsTest.cpp
        api/cli/Utils.cpp

        runtime/distributed/worker/WorkerTest.cpp

        runtime/local/datastructures/CSRMatrixTest.cpp
        runtime/local/datastructures/DenseMatrixTest.cpp
        runtime/local/datastructures/FrameTest.cpp
        runtime/local/datastructures/MatrixTest.cpp
        runtime/local/datastructures/TaskQueueTest.cpp
        
        runtime/local/io/FileMetaDataTest.cpp
        runtime/local/io/ReadCsvTest.cpp
        
        runtime/local/kernels/AggAllTest.cpp
        runtime/local/kernels/AggColTest.cpp
        runtime/local/kernels/AggRowTest.cpp
        runtime/local/kernels/CastObjTest.cpp
        runtime/local/kernels/CastScaTest.cpp
        runtime/local/kernels/CheckEqTest.cpp
        runtime/local/kernels/ColBindTest.cpp
        runtime/local/kernels/CreateFrameTest.cpp
        runtime/local/kernels/DiagVectorTest.cpp
        runtime/local/kernels/EwBinaryMatTest.cpp
        runtime/local/kernels/EwBinaryMatScaTest.cpp
        runtime/local/kernels/EwBinaryScaTest.cpp
        runtime/local/kernels/EwUnaryScaTest.cpp
        runtime/local/kernels/ExtractRowTest.cpp
        runtime/local/kernels/FilterRowTest.cpp
        runtime/local/kernels/GroupJoinTest.cpp
        runtime/local/kernels/HasSpecialValueTest.cpp
        runtime/local/kernels/IsSymmetricTest.cpp
        runtime/local/kernels/NumDistinctApproxTest.cpp
        runtime/local/kernels/MatMulTest.cpp
        runtime/local/kernels/RandMatrixTest.cpp
        runtime/local/kernels/ReadTest.cpp
        runtime/local/kernels/ReplaceTest.cpp
        runtime/local/kernels/SeqTest.cpp
        runtime/local/kernels/SetColLabelsTest.cpp
        runtime/local/kernels/SetColLabelsPrefixTest.cpp
        runtime/local/kernels/SolveTest.cpp
        runtime/local/kernels/SyrkTest.cpp
        runtime/local/kernels/TransposeTest.cpp
        runtime/local/kernels/TriTest.cpp
        
        runtime/local/vectorized/MultiThreadedKernelTest.cpp
)
add_dependencies(run_tests daphnec)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
target_link_libraries(run_tests PRIVATE
    ${dialect_libs}
    DataStructures
    DaphneDSLParser
    IO
    MLIRDaphne
    WorkerImpl
    Proto
    Util
    ${OPENBLAS_LIBRARIES}
)
