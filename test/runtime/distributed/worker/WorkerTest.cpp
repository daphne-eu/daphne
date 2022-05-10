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


#include "runtime/distributed/proto/worker.pb.h"
#include "runtime/distributed/proto/worker.grpc.pb.h"
#include "runtime/distributed/worker/WorkerImpl.h"
#include "runtime/distributed/worker/ProtoDataConverter.h"
#include "runtime/local/kernels/EwBinaryMat.h"
#include "runtime/local/kernels/CheckEq.h"

#include <tags.h>

#include <catch.hpp>

#include <grpcpp/grpcpp.h>
#include <runtime/local/io/File.h>
#include <runtime/local/io/ReadCsv.h>
#include <api/cli/Utils.h>
#include <thread>

const std::string dirPath = "test/runtime/distributed/worker/";

TEST_CASE("Simple distributed worker functionality test", TAG_DISTRIBUTED)
{
    WorkerImpl workerImpl;
    
    WHEN ("Sending a task where no outputs are expected")
    {

        distributed::Task task;
        task.set_mlir_code("func @" + WorkerImpl::DISTRIBUTED_FUNCTION_NAME +
            "() -> () {\n"
            "  \"daphne.return\"() : () -> ()\n"
            "}\n");
        
        grpc::ServerContext context;
        distributed::ComputeResult result;
        grpc::Status status;            
        status = workerImpl.Compute(&context, &task, &result);

        THEN ("No filenames are returned")
        {
            REQUIRE(status.ok());
            REQUIRE(result.outputs_size() == 0);
        }
    }

    WHEN ("Sending simple random generation task")
    {
        distributed::Task task;
        task.set_mlir_code("func @" + WorkerImpl::DISTRIBUTED_FUNCTION_NAME +
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

        grpc::ServerContext context;
        distributed::ComputeResult result;
        grpc::Status status;

        status = workerImpl.Compute(&context, &task, &result);

        THEN ("The filename of the matrix is returned")
        {
            REQUIRE(status.ok());
            REQUIRE(result.outputs_size() == 1);
        }
    }

    WHEN ("Sending a task with a read on the worker") 
    {
        distributed::Task task;
        distributed::WorkData workerInput;
        distributed::StoredData data;
        data.set_filename(dirPath + "mat.csv");
        data.set_num_rows(2);
        data.set_num_cols(4);
        *workerInput.mutable_stored() = data;
        *task.add_inputs() = workerInput;
        task.set_mlir_code(
            "func @" + WorkerImpl::DISTRIBUTED_FUNCTION_NAME +
                "(%mat: !daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64> {\n"
                "  %r = \"daphne.ewAdd\"(%mat, %mat) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>\n"
                "  \"daphne.return\"(%r) : (!daphne.Matrix<?x?xf64>) -> ()\n"
                "}");

        grpc::ServerContext context;
        distributed::ComputeResult result;
        grpc::Status status;
        status = workerImpl.Compute(&context, &task, &result);

        THEN ("A Matrix is returned") {
            REQUIRE(status.ok());
            REQUIRE(result.outputs_size() == 1);
        }
    }

    WHEN ("Sending a task with a read on the worker")
    {
        distributed::Task task;
        distributed::WorkData workerInput;
        distributed::StoredData data;

        auto filename = dirPath + "mat.csv";
        auto rows = 2;
        auto cols = 4;

        data.set_filename(filename);
        data.set_num_rows(rows);
        data.set_num_cols(cols);
        *workerInput.mutable_stored() = data;
        *task.add_inputs() = workerInput;
        task.set_mlir_code(
            "func @" + WorkerImpl::DISTRIBUTED_FUNCTION_NAME +
                "(%mat: !daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64> {\n"
                "  %r = \"daphne.ewAdd\"(%mat, %mat) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>\n"
                "  \"daphne.return\"(%r) : (!daphne.Matrix<?x?xf64>) -> ()\n"
                "}");
        grpc::ServerContext context;
        distributed::ComputeResult result;
        grpc::Status status;
        status = workerImpl.Compute(&context, &task, &result);        

        THEN ("A Matrix is returned with all elements doubled") {
            REQUIRE(status.ok());
            REQUIRE(result.outputs_size() == 1);
            
            distributed::Matrix mat;
            REQUIRE(result.outputs(0).data_case() == distributed::WorkData::kStored);

            
            status = workerImpl.Transfer(&context, &result.outputs(0).stored(), &mat);
            REQUIRE(status.ok());

            DenseMatrix<double> *matOrig = nullptr;
            char delim = ',';
            readCsv(matOrig, filename.c_str(), rows, cols, delim);

            auto *received = DataObjectFactory::create<DenseMatrix<double>>(mat.num_rows(), mat.num_cols(), false);
            ProtoDataConverter::convertFromProto(mat, received);

            DenseMatrix<double> *matOrigTimes2 = nullptr;
            EwBinaryMat<DenseMatrix<double>, DenseMatrix<double>, DenseMatrix<double>>::apply(BinaryOpCode::ADD,
                matOrigTimes2,
                matOrig,
                matOrig,
                nullptr);

            // TODO: epsilon check once it is no longer ensured the same kernel will be used
            REQUIRE(*received == *matOrigTimes2);
        }
    }
}
