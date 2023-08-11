/*
 *  Copyright 2021 The DAPHNE Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */



#include "runtime/distributed/worker/WorkerImpl.h"
#include "runtime/local/kernels/EwBinaryMat.h"
#include "runtime/local/kernels/CheckEq.h"
#include "run_tests.h"

#include <tags.h>

#include <catch.hpp>

#include <runtime/local/io/File.h>
#include <runtime/local/io/ReadCsv.h>
#include <api/cli/Utils.h>
#include <thread>

const std::string dirPath = "test/runtime/distributed/worker/";

TEMPLATE_PRODUCT_TEST_CASE("Simple distributed worker functionality test", TAG_DISTRIBUTED, (DenseMatrix), (double))
{
    auto dctx = setupContextAndLogger();
    using DT = TestType;    
    WorkerImpl workerImpl(user_config);
    
    WHEN ("Sending a task where no outputs are expected")
    {

        std::string task("func.func @" + WorkerImpl::DISTRIBUTED_FUNCTION_NAME +
            "() -> () {\n"
            "  \"daphne.return\"() : () -> ()\n"
            "}\n");
        
        std::vector<WorkerImpl::StoredInfo> inputs, outputs;
        auto status = workerImpl.Compute(&outputs, inputs, task);

        THEN ("No filenames are returned")
        {
            REQUIRE(status.ok());
            REQUIRE(outputs.size() == 0);
        }
    }

    WHEN ("Sending simple random generation task")
    {
        std::string task("func.func @" + WorkerImpl::DISTRIBUTED_FUNCTION_NAME +
            "() -> !daphne.Matrix<?x?xf64> {\n"
            "    %3 = \"daphne.constant\"() {value = 2 : si64} : () -> si64\n"
            "    %4 = \"daphne.cast\"(%3) : (si64) -> index\n"
            "    %5 = \"daphne.constant\"() {value = 3 : si64} : () -> si64\n"
            "    %6 = \"daphne.cast\"(%5) : (si64) -> index\n"
            "    %7 = \"daphne.constant\"() {value = 1.000000e+02 : f64} : () -> f64\n"
            "    %8 = \"daphne.constant\"() {value = 2.000000e+02 : f64} : () -> f64\n"
            "    %9 = \"daphne.constant\"() {value = 1.000000e+00 : f64} : () -> f64\n"
            "    %10 = \"daphne.constant\"() {value = -1 : si64} : () -> si64\n"
            "    %11 = \"daphne.randMatrix\"(%4, %6, %7, %8, %9, %10) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<?x?xf64>"
            "    \"daphne.return\"(%11) : (!daphne.Matrix<?x?xf64>) -> ()\n"
            "  }");

        std::vector<WorkerImpl::StoredInfo> inputs, outputs;
        auto status = workerImpl.Compute(&outputs, inputs, task);

        THEN ("The filename of the matrix is returned")
        {
            REQUIRE(status.ok());
            REQUIRE(outputs.size() == 1);
        }
    }

    WHEN ("Sending a task with a read on the worker") 
    {
        std::vector<WorkerImpl::StoredInfo> inputs, outputs;
        inputs.push_back(WorkerImpl::StoredInfo({dirPath + "mat.csv", 2, 4}));
        std::string task(
            "func.func @" + WorkerImpl::DISTRIBUTED_FUNCTION_NAME +
                "(%mat: !daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64> {\n"
                "  %r = \"daphne.ewAdd\"(%mat, %mat) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>\n"
                "  \"daphne.return\"(%r) : (!daphne.Matrix<?x?xf64>) -> ()\n"
                "}");

        auto status = workerImpl.Compute(&outputs, inputs, task);

        THEN ("A Matrix is returned") {
            REQUIRE(status.ok());
            REQUIRE(outputs.size() == 1);
        }
    }

    WHEN ("Sending a task with a read on the worker")
    {
        std::vector<WorkerImpl::StoredInfo> inputs, outputs;
        
        auto identification = dirPath + "mat.csv";
        size_t rows = 2;
        size_t cols = 4;
        inputs.push_back(WorkerImpl::StoredInfo({identification , rows, cols}));

        std::string task(
            "func.func @" + WorkerImpl::DISTRIBUTED_FUNCTION_NAME +
                "(%mat: !daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64> {\n"
                "  %r = \"daphne.ewAdd\"(%mat, %mat) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>\n"
                "  \"daphne.return\"(%r) : (!daphne.Matrix<?x?xf64>) -> ()\n"
                "}");
        
        auto status = workerImpl.Compute(&outputs, inputs, task);        

        THEN ("A Matrix is returned with all elements doubled") {
            REQUIRE(status.ok());
            REQUIRE(outputs.size() == 1);
            
            Structure *structure;
            
            structure = workerImpl.Transfer(outputs[0]);
            REQUIRE(status.ok());
            
            DT *mat = dynamic_cast<DT*>(structure);
            REQUIRE(mat != nullptr);

            DT *matOrig = nullptr;

            char delim = ',';
            readCsv(matOrig, identification.c_str(), rows, cols, delim);

            DT *matOrigTimes2 = nullptr;
            EwBinaryMat<DT, DT, DT>::apply(BinaryOpCode::ADD,
                matOrigTimes2,
                matOrig,
                matOrig,
                nullptr);

            // TODO: epsilon check once it is no longer ensured the same kernel will be used
            CHECK(*mat == *matOrigTimes2);
        }
    }
}
