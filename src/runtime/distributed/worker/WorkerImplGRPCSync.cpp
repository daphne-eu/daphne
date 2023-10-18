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


#include "WorkerImplGRPCSync.h"

#include <runtime/local/io/DaphneSerializer.h>
#include <runtime/local/datastructures/DataObjectFactory.h>

#include <grpcpp/grpcpp.h>
#include <grpcpp/server_builder.h>

WorkerImplGRPCSync::WorkerImplGRPCSync(const std::string& addr, DaphneUserConfig& _cfg) : WorkerImpl(_cfg)

{
    builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
    builder.RegisterService(this);
    builder.SetMaxReceiveMessageSize(INT_MAX);
    builder.SetMaxSendMessageSize(INT_MAX);
    server = builder.BuildAndStart();
}

void WorkerImplGRPCSync::Wait() {
    server->Wait();
}

grpc::Status WorkerImplGRPCSync::Store(::grpc::ServerContext *context,
                         const ::distributed::Data *request,
                         ::distributed::StoredData *response) 
{
    StoredInfo storedInfo;
    auto buffer = request->bytes().data();
    auto len = request->bytes().size();
    if (DF_Dtype(buffer) == DF_data_t::Value_t) {
        double val = DaphneSerializer<double>::deserialize(buffer);
        storedInfo = WorkerImpl::Store(&val);
    } else {
        Structure *mat = DF_deserialize(buffer, len);
        storedInfo = WorkerImpl::Store(mat);
    }
    response->set_identifier(storedInfo.identifier);
    response->set_num_rows(storedInfo.numRows);
    response->set_num_cols(storedInfo.numCols);
    return ::grpc::Status::OK;
}

grpc::Status WorkerImplGRPCSync::Compute(::grpc::ServerContext *context,
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
    auto respMsg = WorkerImpl::Compute(&outputs, inputs, request->mlir_code());
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

grpc::Status WorkerImplGRPCSync::Transfer(::grpc::ServerContext *context,
                          const ::distributed::StoredData *request,
                         ::distributed::Data *response)
{
    StoredInfo info({request->identifier(), request->num_rows(), request->num_cols()});
    std::vector<char> buffer;
    size_t bufferLength;
    Structure *mat = WorkerImpl::Transfer(info);
    bufferLength = DaphneSerializer<Structure>::serialize(mat, buffer);
    response->set_bytes(buffer.data(), bufferLength);
    return ::grpc::Status::OK;
}