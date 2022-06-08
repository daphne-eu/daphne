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

set(TEST_SOURCES
        run_tests.cpp

        api/cli/algorithms/AlgorithmsTest.cpp
        api/cli/config/ConfigTest.cpp
        api/cli/controlflow/ControlFlowTest.cpp
        api/cli/distributed/DistributedTest.cpp
        api/cli/expressions/MatrixLiteralTest.cpp
        api/cli/expressions/TernaryTest.cpp
        api/cli/functions/FunctionsTest.cpp
        api/cli/functions/RecursiveFunctionsTest.cpp
        api/cli/io/ReadTest.cpp
        api/cli/indexing/IndexingTest.cpp
        api/cli/inference/SparsityTest.cpp
        api/cli/io/ReadTest.cpp
        api/cli/literals/LiteralsTest.cpp
        api/cli/operations/ConstantFoldingTest.cpp
        api/cli/operations/OperationsTest.cpp
        api/cli/parser/ParserTest.cpp
        api/cli/parser/MetaDataParserTest.cpp
        api/cli/scoping/ScopingTest.cpp
        api/cli/scriptargs/ScriptArgsTest.cpp
        api/cli/sql/SQLTest.cpp
        api/cli/syntax/SyntaxTest.cpp
        api/cli/vectorized/MultiThreadedOpsTest.cpp
        api/cli/vectorized/VectorizedPipelineTest.cpp
        api/cli/Utils.cpp
    
        api/python/DaphneLibTest.cpp
        
        parser/config/ConfigParserTest.cpp
    
        runtime/distributed/worker/WorkerTest.cpp
    
        runtime/local/datastructures/CSRMatrixTest.cpp
        runtime/local/datastructures/DenseMatrixTest.cpp
        runtime/local/datastructures/FrameTest.cpp
        runtime/local/datastructures/MatrixTest.cpp
        runtime/local/datastructures/TaskQueueTest.cpp

        runtime/local/io/FileMetaDataTest.cpp
        runtime/local/io/ReadCsvTest.cpp
	runtime/local/io/ReadParquetTest.cpp
	runtime/local/io/ReadMMTest.cpp
	runtime/local/io/WriteDaphneTest.cpp
	runtime/local/io/ReadDaphneTest.cpp

        runtime/local/kernels/AggAllTest.cpp
        runtime/local/kernels/AggColTest.cpp
        runtime/local/kernels/AggRowTest.cpp
        runtime/local/kernels/CartesianTest.cpp
        runtime/local/kernels/CastObjTest.cpp
        runtime/local/kernels/CastScaTest.cpp
        runtime/local/kernels/CheckEqTest.cpp
        runtime/local/kernels/ColBindTest.cpp
        runtime/local/kernels/CreateFrameTest.cpp
        runtime/local/kernels/CTableTest.cpp
        runtime/local/kernels/DiagMatrixTest.cpp
        runtime/local/kernels/DiagVectorTest.cpp
        runtime/local/kernels/DNNPoolingTest.cpp
        runtime/local/kernels/EwBinaryMatTest.cpp
        runtime/local/kernels/EwBinaryObjScaTest.cpp
        runtime/local/kernels/EwBinaryScaTest.cpp
        runtime/local/kernels/EwUnaryScaTest.cpp
        runtime/local/kernels/ExtractColTest.cpp
        runtime/local/kernels/ExtractRowTest.cpp
        runtime/local/kernels/FilterRowTest.cpp
        runtime/local/kernels/GroupJoinTest.cpp
        runtime/local/kernels/GroupTest.cpp
        runtime/local/kernels/HasSpecialValueTest.cpp
        runtime/local/kernels/InnerJoinTest.cpp
        runtime/local/kernels/IsSymmetricTest.cpp
        runtime/local/kernels/NumDistinctApproxTest.cpp
        runtime/local/kernels/MatMulTest.cpp
        runtime/local/kernels/OrderTest.cpp
        runtime/local/kernels/QuantizeTest.cpp
        runtime/local/kernels/RandMatrixTest.cpp
        runtime/local/kernels/ReadTest.cpp
        runtime/local/kernels/ReplaceTest.cpp
        runtime/local/kernels/ReshapeTest.cpp
        runtime/local/kernels/ReverseTest.cpp
        runtime/local/kernels/RowBindTest.cpp
        runtime/local/kernels/SampleTest.cpp
        runtime/local/kernels/SeqTest.cpp
        runtime/local/kernels/SetColLabelsTest.cpp
        runtime/local/kernels/SetColLabelsPrefixTest.cpp
        runtime/local/kernels/SliceColTest.cpp
        runtime/local/kernels/SliceRowTest.cpp
        runtime/local/kernels/SolveTest.cpp
        runtime/local/kernels/SyrkTest.cpp
        runtime/local/kernels/ThetaJoinTest.cpp
        runtime/local/kernels/TransposeTest.cpp
        runtime/local/kernels/TriTest.cpp
        runtime/local/vectorized/MultiThreadedKernelTest.cpp
        runtime/local/kernels/CheckEqApproxTest.cpp

#        runtime/local/kernels/Morphstore/ProjectTest.cpp
)

if(USE_CUDA AND CMAKE_CUDA_COMPILER)
    list(APPEND TEST_SOURCES
            runtime/local/kernels/CUDA/MatMulTest.cpp
            runtime/local/kernels/CUDA_ContextTest.cpp
            runtime/local/kernels/DNNActivationTest.cpp
            runtime/local/kernels/DNNAffineTest.cpp
            runtime/local/kernels/DNNBatchNormTest.cpp
            runtime/local/kernels/DNNConvolutionTest.cpp
            runtime/local/kernels/DNNSoftmaxTest.cpp)
endif()

add_executable(run_tests
    ${TEST_SOURCES}
)
add_dependencies(run_tests daphne DistributedWorker)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
set(LIBS AllKernels ${dialect_libs} DataStructures DaphneDSLParser MLIRDaphne WorkerImpl Proto Util ${OPENBLAS_LIBRARIES} DaphneConfigParser DaphneMetaDataParser)

if(USE_CUDA AND CMAKE_CUDA_COMPILER)
    target_include_directories(run_tests PUBLIC ${CUDAToolkit_INCLUDE_DIRS})

    # search "custom" cudnn lib in CUDA SDK dir
    set(lib_name cudnn)
    find_library(CUDA_${lib_name}_LIBRARY NAMES ${lib_name} HINTS ${CUDAToolkit_LIBRARY_DIR} ENV CUDA_PATH
            PATH_SUFFIXES nvidia/current lib64 lib/x64 lib)

    list(APPEND LIBS CUDAKernels CUDA::cudart CUDA::cublasLt CUDA::cublas CUDA::cusparse ${CUDA_cudnn_LIBRARY})
endif()

target_link_libraries(run_tests PRIVATE ${LIBS})

add_executable(theta_join_test run_tests.cpp runtime/local/kernels/ThetaJoinTest.cpp)
add_dependencies(theta_join_test daphne DistributedWorker)
target_link_libraries(theta_join_test PRIVATE ${LIBS})
target_compile_options(theta_join_test PUBLIC -g -O0)
