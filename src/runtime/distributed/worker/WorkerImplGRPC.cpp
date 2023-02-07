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


#include "WorkerImplGRPC.h"

#include <runtime/distributed/proto/CallData.h>
#include <runtime/local/io/DaphneSerializer.h>
#include <runtime/local/datastructures/DataObjectFactory.h>

#include <grpcpp/grpcpp.h>
#include <grpcpp/server_builder.h>

WorkerImplGRPC::WorkerImplGRPC(std::string addr)
{
    builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
    cq_ = builder.AddCompletionQueue();
    builder.RegisterService(&service_);
    builder.SetMaxReceiveMessageSize(INT_MAX);
    builder.SetMaxSendMessageSize(INT_MAX);
    server = builder.BuildAndStart();
}

void WorkerImplGRPC::Wait() {
    // Spawn a new CallData instance to serve new clients.
    new StoreCallData(this, cq_.get());
    new ComputeCallData(this, cq_.get());
    new TransferCallData(this, cq_.get());
    // new FreeMemCallData(this, cq_.get());
    void* tag;  // uniquely identifies a request.
    bool ok;
    // Block waiting to read the next event from the completion queue. The
    // event is uniquely identified by its tag, which in this case is the
    // memory address of a CallData instance.
    // The return value of Next should always be checked. This return value
    // tells us whether there is any kind of event or cq_ is shutting down.
    while (cq_->Next(&tag, &ok)) {        
        if(ok){         
            // Thread pool ? with caution. For now on each worker only one thread operates (sefe IO).
            // We might need to add locks inside Store/Compute/Transfer methods if we deploy threads
            static_cast<CallData*>(tag)->Proceed();
        } else {
            // TODO maybe handle this internally ?
            delete static_cast<CallData*>(tag);
        }
    }
}

grpc::Status WorkerImplGRPC::StoreGRPC(::grpc::ServerContext *context,
                         const ::distributed::Data *request,
                         ::distributed::StoredData *response) 
{
    StoredInfo storedInfo;
    switch (request->data_case()){
        case distributed::Data::DataCase::kMatrix:
        {
            Structure *mat = nullptr;
            mat = DF_load(request->matrix().bytes().c_str());
            storedInfo = WorkerImpl::Store<Structure>(mat);
            break;
        }
        case distributed::Data::DataCase::kValue:
        {
            auto protoVal = &request->value();
            switch (protoVal->value_case()){
                case distributed::Value::ValueCase::kF64:
                {
                    double val = double(protoVal->f64());
                    storedInfo = WorkerImpl::Store(&val);
                    break;
                }
                default: {
                    throw std::runtime_error(protoVal->value_case() + " not supported atm");
                    break; 
                }
            }
            break;
        }
        default:
            throw std::runtime_error("Store: data not set");
            break;
    }
    response->set_identifier(storedInfo.identifier);
    response->set_num_rows(storedInfo.numRows);
    response->set_num_cols(storedInfo.numCols);
    return ::grpc::Status::OK;
}

grpc::Status WorkerImplGRPC::ComputeGRPC(::grpc::ServerContext *context,
                         const ::distributed::Task *request,
                         ::distributed::ComputeResult *response)
{
    std::vector<StoredInfo> inputs;
    inputs.reserve(request->inputs().size());

    std::vector<StoredInfo> outputs = std::vector<StoredInfo>();
    for (auto input : request->inputs()){
        auto stored = input.stored();
        inputs.push_back(StoredInfo({stored.identifier(), stored.num_rows(), stored.num_cols()}));
    }
    auto respMsg = Compute(&outputs, inputs, request->mlir_code());
    for (auto output : outputs){        
        distributed::WorkData workData;        
        workData.mutable_stored()->set_identifier(output.identifier);
        workData.mutable_stored()->set_num_rows(output.numRows);
        workData.mutable_stored()->set_num_cols(output.numCols);
        *response->add_outputs() = workData;
    }
    if (respMsg.ok())
        return ::grpc::Status::OK;
    else
        return ::grpc::Status(grpc::StatusCode::ABORTED, respMsg.error_message());        
}

grpc::Status WorkerImplGRPC::TransferGRPC(::grpc::ServerContext *context,
                          const ::distributed::StoredData *request,
                         ::distributed::Matrix *response)
{
    StoredInfo info({request->identifier(), request->num_rows(), request->num_cols()});
    void *buffer = nullptr;
    size_t bufferLength;
    Structure *mat = Transfer(info);
    // TODO: Could we avoid if statements here?
    if(auto matDT = dynamic_cast<DenseMatrix<double>*>(mat)){
        buffer = DaphneSerializer<DenseMatrix<double>>::save(matDT, buffer);
        bufferLength = DaphneSerializer<DenseMatrix<double>>::length(matDT);
    }        
    else if(auto matDT = dynamic_cast<DenseMatrix<int64_t>*>(mat)){
        buffer = DaphneSerializer<DenseMatrix<int64_t>>::save(matDT, buffer);
        bufferLength = DaphneSerializer<DenseMatrix<int64_t>>::length(matDT);
    }
    else if(auto matDT = dynamic_cast<CSRMatrix<double>*>(mat)){
        buffer = DaphneSerializer<CSRMatrix<double>>::save(matDT, buffer);
        bufferLength = DaphneSerializer<CSRMatrix<double>>::length(matDT);
    }
    else if(auto matDT = dynamic_cast<CSRMatrix<int64_t>*>(mat)){
        buffer = DaphneSerializer<CSRMatrix<int64_t>>::save(matDT, buffer);
        bufferLength = DaphneSerializer<CSRMatrix<int64_t>>::length(matDT);

    } else 
        std::runtime_error("Type is not supported atm");
    response->set_bytes(buffer, bufferLength);
    return ::grpc::Status::OK;
}