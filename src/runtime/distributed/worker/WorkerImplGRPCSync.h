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

#ifndef SRC_RUNTIME_DISTRIBUTED_WORKER_WORKERIMPLGRPCSYNC_H
#define SRC_RUNTIME_DISTRIBUTED_WORKER_WORKERIMPLGRPCSYNC_H

#include "WorkerImpl.h"

#include <grpcpp/grpcpp.h>
#include <grpcpp/server_builder.h>
#include "runtime/distributed/proto/worker.pb.h"
#include "runtime/distributed/proto/worker.grpc.pb.h"

#include <runtime/local/io/DaphneSerializer.h>

class WorkerImplGRPCSync : public WorkerImpl, public distributed::Worker::Service
{
private:
    grpc::ServerBuilder builder;
    std::unique_ptr<grpc::Server> server;
    // Store in chunks
    std::unique_ptr<DaphneDeserializerChunks<Structure>> deserializer;
    std::unique_ptr<DaphneDeserializerChunks<Structure>::Iterator> deserializerIter;
    Structure *mat;

public:
    explicit WorkerImplGRPCSync(const std::string& addr, DaphneUserConfig& _cfg);
    void Wait() override;
    grpc::Status Store(::grpc::ServerContext *context,
                         ::grpc::ServerReader<::distributed::Data>* reader,
                         ::distributed::StoredData *response) override;
    grpc::Status Compute(::grpc::ServerContext *context,
                         const ::distributed::Task *request,
                         ::distributed::ComputeResult *response) override;
    grpc::Status Transfer(::grpc::ServerContext *context,
                          const ::distributed::StoredData *request,
                         ::distributed::Data *response) override;

    template<class DT>
    DT* CreateMatrix(const ::distributed::Data *mat);
};

#endif //SRC_RUNTIME_DISTRIBUTED_WORKER_WORKERIMPLGRPCSYNC_H
