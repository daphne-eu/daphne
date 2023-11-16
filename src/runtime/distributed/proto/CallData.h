
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

#pragma once

#include <runtime/distributed/worker/WorkerImplGRPCAsync.h>
#include <runtime/distributed/proto/worker.pb.h>
#include <runtime/distributed/proto/worker.grpc.pb.h>

class CallData
{
public:
    virtual void Proceed(bool ok) = 0;
    virtual ~CallData() = default;
};
class StoreCallData final : public CallData
{
public:
    StoreCallData(WorkerImplGRPCAsync *worker_, grpc::ServerCompletionQueue *scq, grpc::ServerCompletionQueue *cq)
        : worker(worker_), service_(&worker_->service_), scq_(scq), cq_(cq), stream_(&ctx_), responder_(&ctx_), status_(CREATE)
    {
        // Invoke the serving logic right away.
        Proceed(true);
    }

    void Proceed(bool ok) override;

private:
    WorkerImplGRPCAsync *worker;
    distributed::Worker::AsyncService *service_;
    // The producer-consumer queue where for asynchronous server notifications.
    grpc::ServerCompletionQueue *scq_;
    grpc::ServerCompletionQueue *cq_;
    grpc::ServerContext ctx_;
    // What we get from the client.
    distributed::Data data;
    // What we send back to the client.
    distributed::StoredData storedData;
    // The means to get back to the client.
    grpc::ServerAsyncReader<distributed::StoredData, distributed::Data> stream_;
    grpc::ServerAsyncResponseWriter<distributed::StoredData> responder_;

    // Let's implement a tiny state machine with the following states.
    enum CallStatus
    {
        CREATE,
        PROCESS,
        FINISH
    };
    CallStatus status_; // The current serving state.
};
class ComputeCallData final : public CallData
{
public:
    ComputeCallData(WorkerImplGRPCAsync *worker_, grpc::ServerCompletionQueue *cq)
        : worker(worker_), service_(&worker_->service_), cq_(cq), responder_(&ctx_), status_(CREATE)
    {
        // Invoke the serving logic right away.
        Proceed(true);
    }

    void Proceed(bool ok) override;

private:
    WorkerImplGRPCAsync *worker;
    distributed::Worker::AsyncService *service_;
    // The producer-consumer queue where for asynchronous server notifications.
    grpc::ServerCompletionQueue *cq_;
    grpc::ServerContext ctx_;
    // What we get from the client.
    distributed::Task task;
    // What we send back to the client.
    distributed::ComputeResult result;
    // The means to get back to the client.
    grpc::ServerAsyncResponseWriter<distributed::ComputeResult> responder_;

    // Let's implement a tiny state machine with the following states.
    enum CallStatus
    {
        CREATE,
        PROCESS,
        FINISH
    };
    CallStatus status_; // The current serving state.
};

class TransferCallData final : public CallData
{
public:
    TransferCallData(WorkerImplGRPCAsync *worker_, grpc::ServerCompletionQueue *cq)
        : worker(worker_), service_(&worker_->service_), cq_(cq), responder_(&ctx_), status_(CREATE)
    {
        // Invoke the serving logic right away.
        Proceed(true);
    }
    void Proceed(bool ok) override;
private:
    WorkerImplGRPCAsync *worker;
    distributed::Worker::AsyncService *service_;
    // The producer-consumer queue where for asynchronous server notifications.
    grpc::ServerCompletionQueue *cq_;
    grpc::ServerContext ctx_;
    // What we get from the client.
    distributed::StoredData storedData;
    // What we send back to the client.
    distributed::Data data;
    // The means to get back to the client.
    grpc::ServerAsyncResponseWriter<distributed::Data> responder_;

    // Let's implement a tiny state machine with the following states.
    enum CallStatus
    {
        CREATE,
        PROCESS,
        FINISH
    };
    CallStatus status_; // The current serving state.
};

// class FreeMemCallData final : public CallData
// {
//     public:
//         FreeMemCallData(WorkerImplGRPCAsync *worker_, grpc::ServerCompletionQueue *cq)
//             : worker(worker_), service_(&worker_->service_), cq_(cq), responder_(&ctx_), status_(CREATE)
//         {
//             // Invoke the serving logic right away.
//             Proceed();
//         }
//         void Proceed() override;
//     private:
//         WorkerImplGRPCAsync *worker;
//         distributed::Worker::AsyncService *service_;
//         // The producer-consumer queue where for asynchronous server notifications.
//         grpc::ServerCompletionQueue *cq_;
//         grpc::ServerContext ctx_;
        
//         distributed::StoredData storedData;
//         distributed::Empty emptyMessage;
//         grpc::ServerAsyncResponseWriter<distributed::Empty> responder_;

//     enum CallStatus
//     {
//         CREATE,
//         PROCESS,
//         FINISH
//     };
//     CallStatus status_; // The current serving state.
// };
