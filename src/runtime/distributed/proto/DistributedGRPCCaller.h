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

#ifndef SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_DISTRIBUTEDCALLER_H
#define SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_DISTRIBUTEDCALLER_H
#include <grpcpp/grpcpp.h>
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
class DistributedGRPCCaller {
private:
    
    /**
     * @brief   Map to keep channels.
     *          We declare channels as inline static in order to reuse them
     *          across different Distributed kernels. 
     *          TODO: Move channels map to DistributedContext.
     */
    inline static std::map<std::string, std::shared_ptr<grpc::Channel>> channels;
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
        
        StoredInfo storedInfo;
        ReturnType result;
    };
    int callCounter = 0;
    grpc::CompletionQueue cq_;

public:
    DistributedGRPCCaller() {};
    ~DistributedGRPCCaller() {};
    
    /**
    * @brief Enqueues an asynchronous Store call to be executed.     
    * 
    * @param  workerAddr An address (or channel) to make the call
    * @param  StoredInfo An StoredInfo type returned when call response is ready
    * @param  arg Argument passed to the asynchronous call
    */
    void asyncStoreCall(
        const std::shared_ptr<grpc::Channel> &channel,
        const StoredInfo &storedInfo,
        const Argument &arg
        )
    {
        AsyncClientCall *call = new AsyncClientCall;
        call->storedInfo = storedInfo;

        auto stub = distributed::Worker::NewStub(channel);
        auto response_reader = stub->AsyncStore(&call->context_, arg, &cq_);
        
        response_reader->Finish(&call->result, &call->status, (void*)call);
        callCounter++;
    }
    void asyncStoreCall(
        const std::string &workerAddr,
        const StoredInfo &storedInfo,
        const distributed::Data &arg
        )
    {
        auto channel = GetOrCreateChannel(workerAddr);
        asyncStoreCall(channel, storedInfo, arg);
    }
    /**
    * @brief Enqueues an asynchronous Compute call to be executed.     
    * 
    * @param  workerAddr An address (or channel) to make the call
    * @param  StoredInfo An StoredInfo type returned when call response is ready
    * @param  arg Argument passed to the asynchronous call
    */
    void asyncComputeCall(
        const std::shared_ptr<grpc::Channel> &channel,
        const StoredInfo &storedInfo,
        const Argument &arg
        )
    {
        AsyncClientCall *call = new AsyncClientCall;
        call->storedInfo = storedInfo;

        auto stub = distributed::Worker::NewStub(channel);
        auto response_reader = stub->AsyncCompute(&call->context_, arg, &cq_);
        
        response_reader->Finish(&call->result, &call->status, (void*)call);
        callCounter++;
    }
    void asyncComputeCall(
        const std::string &workerAddr,
        const StoredInfo &storedInfo,
        const Argument &arg
        )
    {
        auto channel = GetOrCreateChannel(workerAddr);
        asyncComputeCall(channel, storedInfo, arg);
    }
    /**
    * @brief Enqueues an asynchronous Transfer call to be executed.     
    * 
    * @param  workerAddr An address (or channel) to make the call
    * @param  StoredInfo An StoredInfo type returned when call response is ready
    * @param  arg Argument passed to the asynchronous call
    */
    void asyncTransferCall(
        const std::shared_ptr<grpc::Channel> &channel,
        const StoredInfo &storedInfo,
        const Argument &arg
        )
    {
        AsyncClientCall *call = new AsyncClientCall;
        call->storedInfo = storedInfo;

        auto stub = distributed::Worker::NewStub(channel);
        auto response_reader = stub->AsyncTransfer(&call->context_, arg, &cq_);
        
        response_reader->Finish(&call->result, &call->status, (void*)call);
        callCounter++;
    }
    void asyncTransferCall(
        const std::string &workerAddr,
        const StoredInfo &storedInfo,
        const Argument &arg
        )
    {
        auto channel = GetOrCreateChannel(workerAddr);
        asyncTransferCall(channel, storedInfo, arg);
    }
    
    /**
    * @brief Enqueues an asynchronous FreeMem call to be executed.     
    * 
    * @param  workerAddr An address (or channel) to make the call
    * @param  StoredInfo An StoredInfo type returned when call response is ready
    * @param  arg Argument passed to the asynchronous call
    */
    void asyncFreeMemCall(
        const std::shared_ptr<grpc::Channel> &channel,
        const StoredInfo &storedInfo,
        const Argument &arg
        )
    {
        AsyncClientCall *call = new AsyncClientCall;
        call->storedInfo = storedInfo;

        auto stub = distributed::Worker::NewStub(channel);
        auto response_reader = stub->AsyncFreeMem(&call->context_, arg, &cq_);
        
        response_reader->Finish(&call->result, &call->status, (void*)call);
        callCounter++;
    }
    void asyncFreeMemCall(
        const std::string &workerAddr,
        const StoredInfo &storedInfo,
        const Argument &arg
        )
    {
        auto channel = GetOrCreateChannel(workerAddr);
        asyncFreeMemCall(channels, storedInfo, arg);
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
        if (!(ok && call->status.ok())){
            throw std::runtime_error(
                call->status.error_message()
            );
        }
        ResultData ret({call->storedInfo, call->result});
        delete call;
        return ret;
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
    std::shared_ptr<grpc::Channel> GetOrCreateChannel(const std::string &workerAddr) {
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

#endif //SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_DISTRIBUTEDCALLER_H