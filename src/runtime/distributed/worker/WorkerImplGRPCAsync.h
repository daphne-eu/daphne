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

#ifndef SRC_RUNTIME_DISTRIBUTED_WORKER_WORKERIMPLGRPCASYNC_H
#define SRC_RUNTIME_DISTRIBUTED_WORKER_WORKERIMPLGRPCASYNC_H

#include "WorkerImpl.h"

#include <grpcpp/grpcpp.h>
#include <grpcpp/server_builder.h>
#include "runtime/distributed/proto/worker.pb.h"
#include "runtime/distributed/proto/worker.grpc.pb.h"


class WorkerImplGRPCAsync : public WorkerImpl 
{
private:
    std::unique_ptr<grpc::ServerCompletionQueue> cq_;
    grpc::ServerBuilder builder;
    std::unique_ptr<grpc::Server> server;
public:
    explicit WorkerImplGRPCAsync(const std::string& addr, DaphneUserConfig& _cfg);
    void Wait() override;

    grpc::Status StoreGRPC(::grpc::ServerContext *context,
                         const ::distributed::Data *request,
                         ::distributed::StoredData *response) ;
    grpc::Status ComputeGRPC(::grpc::ServerContext *context,
                         const ::distributed::Task *request,
                         ::distributed::ComputeResult *response);
    grpc::Status TransferGRPC(::grpc::ServerContext *context,
                          const ::distributed::StoredData *request,
                         ::distributed::Data *response) ;

    distributed::Worker::AsyncService service_;
};

#endif //SRC_RUNTIME_DISTRIBUTED_WORKER_WORKERIMPLGRPCASYNC_H
