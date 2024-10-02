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

#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/io/DaphneSerializer.h>

#include <grpcpp/grpcpp.h>
#include <grpcpp/server_builder.h>

#if USE_HDFS
#include <runtime/local/io/HDFS/ReadDaphneHDFS.h>
#include <runtime/local/io/HDFS/ReadHDFSCsv.h>
#include <runtime/local/io/HDFS/WriteDaphneHDFS.h>
#include <runtime/local/io/HDFS/WriteHDFSCsv.h>
#include <runtime/local/kernels/CreateHDFSContext.h>
#endif
#include <util/KernelDispatchMapping.h>
#include <util/Statistics.h>
#include <util/StringRefCount.h>

WorkerImplGRPCSync::WorkerImplGRPCSync(const std::string &addr, DaphneUserConfig &_cfg)
    : WorkerImpl(_cfg)

{
    builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
    builder.RegisterService(this);
    builder.SetMaxReceiveMessageSize(INT_MAX);
    builder.SetMaxSendMessageSize(INT_MAX);
    server = builder.BuildAndStart();
}

void WorkerImplGRPCSync::Wait() { server->Wait(); }

grpc::Status WorkerImplGRPCSync::Store(::grpc::ServerContext *context,
                                       ::grpc::ServerReader<::distributed::Data> *reader,
                                       ::distributed::StoredData *response) {
    StoredInfo storedInfo;
    distributed::Data data;
    reader->Read(&data);

    auto buffer = data.bytes().data();
    auto len = data.bytes().size();
    if (DF_Dtype(buffer) == DF_data_t::Value_t) {
        double val = DaphneSerializer<double>::deserialize(buffer);
        storedInfo = WorkerImpl::Store(&val);

        response->set_identifier(storedInfo.identifier);
        response->set_num_rows(storedInfo.numRows);
        response->set_num_cols(storedInfo.numCols);
    } else {
        deserializer.reset(new DaphneDeserializerChunks<Structure>(&mat, len));
        deserializerIter.reset(new DaphneDeserializerChunks<Structure>::Iterator(deserializer->begin()));

        (*deserializerIter)->second->resize(len);
        (*deserializerIter)->first = len;

        if ((*deserializerIter)->second->size() < len)
            (*deserializerIter)->second->resize(len);
        (*deserializerIter)->second->assign(static_cast<const char *>(buffer), static_cast<const char *>(buffer) + len);

        // advance iterator, this also partially deserializes
        ++(*deserializerIter);
        while (reader->Read(&data)) {
            buffer = data.bytes().data();
            len = data.bytes().size();
            (*deserializerIter)->first = len;
            if ((*deserializerIter)->second->size() < len)
                (*deserializerIter)->second->resize(len);
            (*deserializerIter)
                ->second->assign(static_cast<const char *>(buffer), static_cast<const char *>(buffer) + len);

            // advance iterator, this also partially deserializes
            ++(*deserializerIter);
        }
        storedInfo = WorkerImpl::Store(mat);
        response->set_identifier(storedInfo.identifier);
        response->set_num_rows(storedInfo.numRows);
        response->set_num_cols(storedInfo.numCols);
    }
    return ::grpc::Status::OK;
}

grpc::Status WorkerImplGRPCSync::Compute(::grpc::ServerContext *context, const ::distributed::Task *request,
                                         ::distributed::ComputeResult *response) {
    std::vector<StoredInfo> inputs;
    inputs.reserve(request->inputs().size());

    std::vector<StoredInfo> outputs = std::vector<StoredInfo>();
    for (auto input : request->inputs()) {
        auto stored = input.stored();
        inputs.push_back(StoredInfo({stored.identifier(), stored.num_rows(), stored.num_cols()}));
    }
    auto respMsg = WorkerImpl::Compute(&outputs, inputs, request->mlir_code());
    for (auto output : outputs) {
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

grpc::Status WorkerImplGRPCSync::Transfer(::grpc::ServerContext *context, const ::distributed::StoredData *request,
                                          ::distributed::Data *response) {
    StoredInfo info({request->identifier(), request->num_rows(), request->num_cols()});
    std::vector<char> buffer;
    size_t bufferLength;
    Structure *mat = WorkerImpl::Transfer(info);
    bufferLength = DaphneSerializer<Structure>::serialize(mat, buffer);
    response->set_bytes(buffer.data(), bufferLength);
    return ::grpc::Status::OK;
}

#if USE_HDFS
grpc::Status WorkerImplGRPCSync::ReadHDFS(::grpc::ServerContext *context, const ::distributed::HDFSFile *request,
                                          ::distributed::StoredData *response) {
    DaphneContext ctx(cfg, KernelDispatchMapping::instance(), Statistics::instance(), StringRefCounter::instance());
    createHDFSContext(&ctx);
    DenseMatrix<double> *res =
        DataObjectFactory::create<DenseMatrix<double>>(request->num_rows(), request->num_cols(), false);
    if (request->filename().find("csv") != std::string::npos)
        readHDFSCsv(res, request->filename().c_str(), request->num_rows(), request->num_cols(), ',', &ctx,
                    request->start_row());
    else if (request->filename().find("dbdf") != std::string::npos)
        readDaphneHDFS(res, request->filename().c_str(), &ctx, request->start_row());
    auto storedInfo = WorkerImpl::Store(dynamic_cast<Structure *>(res));

    response->set_identifier(storedInfo.identifier);
    response->set_num_rows(storedInfo.numRows);
    response->set_num_cols(storedInfo.numCols);
    return ::grpc::Status::OK;
}

grpc::Status WorkerImplGRPCSync::WriteHDFS(::grpc::ServerContext *context, const ::distributed::HDFSWriteInfo *request,
                                           ::distributed::Empty *response) {
    DaphneContext ctx(cfg, KernelDispatchMapping::instance(), Statistics::instance(), StringRefCounter::instance());
    createHDFSContext(&ctx);
    StoredInfo si({request->matrix().identifier(), request->matrix().num_rows(), request->matrix().num_cols()});
    auto mat = dynamic_cast<DenseMatrix<double> *>(WorkerImpl::Transfer(si));
    if (request->dirname().find("csv") != std::string::npos)
        writeHDFSCsv(mat, request->dirname().c_str(), &ctx);
    else if (request->dirname().find("dbdf") != std::string::npos)
        writeDaphneHDFS(mat, request->dirname().c_str(), &ctx);
    return ::grpc::Status::OK;
}
#endif