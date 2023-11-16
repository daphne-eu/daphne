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


#include "WorkerImplGRPCAsync.h"

#include <runtime/distributed/proto/CallData.h>
#include <runtime/local/io/DaphneSerializer.h>
#include <runtime/local/datastructures/DataObjectFactory.h>

#include <grpcpp/grpcpp.h>
#include <grpcpp/server_builder.h>

WorkerImplGRPCAsync::WorkerImplGRPCAsync(const std::string& addr, DaphneUserConfig& _cfg) : WorkerImpl(_cfg), stream_(&ctx_), responder_(&ctx_)
{
    builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
    cq_ = builder.AddCompletionQueue();
    builder.RegisterService(&service_);
    builder.SetMaxReceiveMessageSize(INT_MAX);
    builder.SetMaxSendMessageSize(INT_MAX);
    server = builder.BuildAndStart();
}

void WorkerImplGRPCAsync::Wait() {
    // Spawn a new CallData instance to serve new clients.
    new StoreCallData(this, cq_.get(), cq_.get());
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
        static_cast<CallData*>(tag)->Proceed(ok);
    }
}

grpc::Status WorkerImplGRPCAsync::StoreGRPC(::grpc::ServerContext *context,
                         const ::distributed::Data *request,
                         ::distributed::StoredData *response) 
{
    StoredInfo storedInfo;
    size_t bufferLength = request->bytes().size();
    // If deserializer's chunk size is 0, it means we just begin receiving data
    if (bufferLength == 0 && this->isFirstChunk == true)
        return grpc::Status::OK;
    // The first actual message received, is the chunk size.
    if (isFirstChunk){
        this->isFirstChunk = false;
        size_t chunkSize;
        auto buffer = request->bytes().data();
        std::copy(buffer, buffer + sizeof(size_t), reinterpret_cast<char*>(&chunkSize));
        deserializer->chunkSize = chunkSize;
        return grpc::Status::OK;
    }

    // Handle value case
    if (*deserializerIter == deserializer->begin() && DF_Dtype(request->bytes().data()) == DF_data_t::Value_t) {
        double val = DaphneSerializer<double>::deserialize(request->bytes().data());
        storedInfo = WorkerImpl::Store(&val);
        response->set_identifier(storedInfo.identifier);
        response->set_num_rows(storedInfo.numRows);
        response->set_num_cols(storedInfo.numCols);
        return ::grpc::Status::OK;    
    } else {
        // partially deserialize next
        (*deserializerIter)->first = bufferLength;
        if ((*deserializerIter)->second->size() < bufferLength)
            (*deserializerIter)->second->resize(bufferLength);
        (*deserializerIter)->second->assign(static_cast<const char*>(request->bytes().data()), static_cast<const char*>(request->bytes().data()) + bufferLength);
        
        // advance iterator, this also partially deserializes
        ++(*deserializerIter);

        // response if we completed
        if (*deserializerIter == deserializer->end()) {
            storedInfo = WorkerImpl::Store(mat);
            response->set_identifier(storedInfo.identifier);
            response->set_num_rows(storedInfo.numRows);
            response->set_num_cols(storedInfo.numCols);
            return ::grpc::Status::OK;    
        }
    }
    return ::grpc::Status::OK;
}

void WorkerImplGRPCAsync::PrepareStoreGRPC() {
    this->isFirstChunk = true;
    size_t chunkSize = DaphneSerializer<Structure>::HEADER_BUFFER_SIZE + 1; // This will be updated when the first request happens and master node sends the chunk size
    deserializer.reset(new DaphneDeserializerChunks<Structure>(&mat, chunkSize));
    deserializerIter.reset(new DaphneDeserializerChunks<Structure>::Iterator(deserializer->begin()));    
    (*deserializerIter)->second->resize(chunkSize);
}

grpc::Status WorkerImplGRPCAsync::ComputeGRPC(::grpc::ServerContext *context,
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

grpc::Status WorkerImplGRPCAsync::TransferGRPC(::grpc::ServerContext *context,
                          const ::distributed::StoredData *request,
                         ::distributed::Data *response)
{
    StoredInfo info({request->identifier(), request->num_rows(), request->num_cols()});
    std::vector<char> buffer;
    size_t bufferLength;
    Structure *mat = Transfer(info);
    bufferLength = DaphneSerializer<Structure>::serialize(mat, buffer);
    response->set_bytes(buffer.data(), bufferLength);
    return ::grpc::Status::OK;
}
