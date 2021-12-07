/*
 * Copyright 2021 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SRC_RUNTIME_DISTRIBUTED_WORKER_DISTRIBUTEDCALLER_H
#define SRC_RUNTIME_DISTRIBUTED_WORKER_DISTRIBUTEDCALLER_H

#include <runtime/local/datastructures/Handle.h>
#include <runtime/distributed/proto/worker.pb.h>
#include <runtime/distributed/proto/worker.grpc.pb.h>

// ****************************************************************************
// Class for async communication
// ****************************************************************************

/**
* @brief Creates an object allowing asynchronous
*        communication between daphne and workers
* 
* @tparam  StoredInfo An object/value holding information related to a specific call, returned upon completition
* @tparam  Argument The class used as argument for the call
* @tparam  ReturnType The result class returned by the call
*/
template<class StoredInfo, class Argument, class ReturnType>
class DistributedCaller {
private:
    
    /**
     * @brief Map to keep channels
     */
    std::map<std::string, std::shared_ptr<grpc::Channel>> channels;
    struct ResultData 
    {
        // Contains struct StoredInfo, that was passed when call was made
        StoredInfo storedInfo;
        // Contains the actual result of the call
        ReturnType result;
    };
    struct AsyncClientCall
    {
        grpc::ClientContext context_;
        grpc::Status status;
        
        const DistributedIndex *ix;
        StoredInfo storedInfo;
        ReturnType result;
    };
    int callCounter = 0;
    grpc::CompletionQueue cq_;


    // Specialized functions to match appropriate async call
    std::unique_ptr<grpc::ClientAsyncResponseReader<distributed::StoredData>> MakeAsyncCall(
        std::unique_ptr<distributed::Worker::Stub> &stub, 
        grpc::ClientContext &context, 
        const distributed::Matrix &arg) {        
            return stub->AsyncStore(&context, arg, &cq_);
    }

    std::unique_ptr<grpc::ClientAsyncResponseReader<distributed::ComputeResult>> MakeAsyncCall(
        std::unique_ptr<distributed::Worker::Stub> &stub, 
        grpc::ClientContext &context, 
        const distributed::Task &arg) {
            return stub->AsyncCompute(&context, arg, &cq_);
    }

    std::unique_ptr<grpc::ClientAsyncResponseReader<distributed::Matrix>> MakeAsyncCall(
        std::unique_ptr<distributed::Worker::Stub> &stub, 
        grpc::ClientContext &context, 
        const distributed::StoredData &arg) {
            return stub->AsyncTransfer(&context, arg, &cq_);
    }
public:
    DistributedCaller() {};
    ~DistributedCaller() {};

    /**
    * @brief Enqueues an asynchronous Transfer call to be executed
    * 
    * @param  workerAddr An address to make the call
    * @param  StoredInfo An StoredInfo type returned when call response is ready
    * @param  Argument Argument passed to the asynchronous call
    */
    void addAsyncCall(
            std::string workerAddr,
            StoredInfo storedInfo,
            const Argument arg
            )
    {
        AsyncClientCall *call = new AsyncClientCall;

        call->storedInfo = storedInfo;
        
        auto channel = GetOrCreateChannel(workerAddr);
        auto stub = distributed::Worker::NewStub(channel);

        auto response_reader = MakeAsyncCall(stub, call->context_, arg);
        response_reader->Finish(&call->result, &call->status, (void*)call);
        callCounter++;
    }
    /**
    * @brief Enqueues an asynchronous call to be executed. 
    *        Matches the appropriate function call (Store/Compute/Transfer) based on template parameters.
    * 
    * @param  workerAddr An address to make the call
    * @param  StoredInfo An StoredInfo type returned when call response is ready
    * @param  Argument Argument passed to the asynchronous call
    */
    void addAsyncCall(
            std::shared_ptr<grpc::Channel> channel,
            StoredInfo storedInfo,
            const Argument arg
            )
    {
        AsyncClientCall *call = new AsyncClientCall;

        call->storedInfo = storedInfo;
        
        auto stub = distributed::Worker::NewStub(channel);

        auto response_reader = MakeAsyncCall(stub, call->context_, arg);
        response_reader->Finish(&call->result, &call->status, (void*)call);
        callCounter++;
    }

    /**
    * @brief    Get the next available result from the queue of asynchronous calls
    * @result   A struct with two fields. First field is "StoredInfo" struct passed when the call was enqueued
    *           and second field is "ReturnType" result of the call
    */
    ResultData getNextResult() {
        void *got_tag;
        bool ok = false;
        cq_.Next(&got_tag, &ok);
        callCounter--;
        AsyncClientCall *call = static_cast<AsyncClientCall*>(got_tag);
        if (!ok){
            throw std::runtime_error(
                call->status.error_message()
            );
        }                
        return ResultData({call->storedInfo, call->result});
    };

    /**
    * @brief Returns True if there are no more async calls to wait
    */
    bool isQueueEmpty() {
        return (callCounter == 0);
    };

    /**
    * @brief Get or Create a new channel for an address
    * 
    * @param workerAddr The address to get the channel for
    */
    std::shared_ptr<grpc::Channel> GetOrCreateChannel(std::string workerAddr) {
        if (channels.count(workerAddr))
            return channels.at(workerAddr);
        else { 
            // Create and store channel
            grpc::ChannelArguments ch_args;
            ch_args.SetMaxSendMessageSize(-1);
            ch_args.SetMaxReceiveMessageSize(-1);
            auto channel = grpc::CreateCustomChannel(workerAddr, grpc::InsecureChannelCredentials(), ch_args);
            channels.insert({workerAddr, channel});
            return channel;
        }
    };

    
};

#endif //SRC_RUNTIME_DISTRIBUTED_WORKER_DISTRIBUTEDCALLER_H