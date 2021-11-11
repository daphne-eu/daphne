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
    auto addr = "0.0.0.0:50051";
    WorkerImpl workerImpl;
    auto server = startDistributedWorker(addr, &workerImpl);
    std::thread thread1 = std::thread(&WorkerImpl::HandleRpcs, &workerImpl);
    
    auto channel = grpc::CreateChannel(addr, grpc::InsecureChannelCredentials());
    auto stub = distributed::Worker::NewStub(channel);
    
    // WHEN ("Sending a task where no outputs are expected")
    {
        struct AsyncCallClient
        {
            grpc::ClientContext context;
            distributed::ComputeResult result;
            grpc::Status status;            
        };
        grpc::CompletionQueue cq;

        distributed::Task task;
        task.set_mlir_code("func @" + WorkerImpl::DISTRIBUTED_FUNCTION_NAME +
            "() -> () {\n"
            "  \"daphne.return\"() : () -> ()\n"
            "}\n");
        AsyncCallClient *call = new AsyncCallClient;
        auto response_reader = stub->AsyncCompute(&call->context, task, &cq);
        response_reader->Finish(&call->result, &call->status, (void*)call);

        void *got_tag;
        bool ok ;

        cq.Next(&got_tag, &ok);

        AsyncCallClient *resultCall = static_cast<AsyncCallClient *>(got_tag);

        THEN ("No filenames are returned")
        {
            REQUIRE(resultCall->status.ok());
            REQUIRE(resultCall->result.outputs_size() == 0);
        }
    }

    // WHEN ("Sending simple random generation task")
    {
        struct AsyncCallClient
        {
            grpc::ClientContext context;
            distributed::ComputeResult result;
            grpc::Status status;            
        };
        grpc::CompletionQueue cq;
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
        AsyncCallClient *call = new AsyncCallClient;
        auto response_reader = stub->AsyncCompute(&call->context, task, &cq);
        response_reader->Finish(&call->result, &call->status, (void*)call);

        void *got_tag;
        bool ok ;
        cq.Next(&got_tag, &ok);

        AsyncCallClient *resultCall = static_cast<AsyncCallClient *>(got_tag);

        THEN ("The filename of the matrix is returned")
        {
            REQUIRE(resultCall->status.ok());
            REQUIRE(resultCall->result.outputs_size() == 1);
        }
    }

    // WHEN ("Sending a task with a read on the worker") 
    {
        struct AsyncCallClient
        {
            grpc::ClientContext context;
            distributed::ComputeResult result;
            grpc::Status status;            
        };
        grpc::CompletionQueue cq;
        
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
        AsyncCallClient *call = new AsyncCallClient;
        auto response_reader = stub->AsyncCompute(&call->context, task, &cq);
        response_reader->Finish(&call->result, &call->status, (void*)call);
        
        void *got_tag;
        bool ok ;
        cq.Next(&got_tag, &ok);        
        AsyncCallClient *resultCall = static_cast<AsyncCallClient *>(got_tag);

        THEN ("A Matrix is returned") {
            REQUIRE(resultCall->status.ok());
            REQUIRE(resultCall->result.outputs_size() == 1);
        }
    }

    // WHEN ("Sending a task with a read on the worker")
    {
        struct AsyncCallClient
        {
            grpc::ClientContext context;
            distributed::ComputeResult result;
            grpc::Status status;            
        };
        grpc::CompletionQueue cq;

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

        AsyncCallClient *call = new AsyncCallClient;
        auto response_reader = stub->AsyncCompute(&call->context, task, &cq);
        response_reader->Finish(&call->result, &call->status, (void*)call);
        
        void *got_tag;
        bool ok ;
        cq.Next(&got_tag, &ok);
        AsyncCallClient *resultCall = static_cast<AsyncCallClient *>(got_tag);

        THEN ("A Matrix is returned with all elements doubled") {
            REQUIRE(resultCall->status.ok());
            REQUIRE(resultCall->result.outputs_size() == 1);
            struct AsyncTransferCallClient
                    {
                        grpc::ClientContext transferContext;
                        distributed::Matrix mat;
                        grpc::Status status;            
                    };
            REQUIRE(resultCall->result.outputs(0).data_case() == distributed::WorkData::kStored);

            AsyncTransferCallClient *transferCall = new AsyncTransferCallClient;
            auto response_transfer_reader = stub->AsyncTransfer(&transferCall->transferContext, resultCall->result.outputs(0).stored(), &cq);
            response_transfer_reader->Finish(&transferCall->mat, &transferCall->status, (void*)transferCall);

            void *got_tag;
            bool ok ;
            cq.Next(&got_tag, &ok);
            AsyncTransferCallClient *resultTransferCall = static_cast<AsyncTransferCallClient *>(got_tag);
            
            REQUIRE(resultTransferCall->status.ok());

            DenseMatrix<double> *matOrig = nullptr;
            struct File *file = openFile(filename.c_str());
            char delim = ',';
            readCsv(matOrig, file, rows, cols, delim);

            auto *received = DataObjectFactory::create<DenseMatrix<double>>(resultTransferCall->mat.num_rows(), resultTransferCall->mat.num_cols(), false);
            ProtoDataConverter::convertFromProto(resultTransferCall->mat, received);

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
    server->Shutdown();
    workerImpl.cq_->Shutdown();
    thread1.join();
}