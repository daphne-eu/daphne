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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_BROADCAST_H
#define SRC_RUNTIME_LOCAL_KERNELS_BROADCAST_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Handle.h>

#include <runtime/distributed/proto/worker.pb.h>
#include <runtime/distributed/proto/worker.grpc.pb.h>
#include <runtime/distributed/worker/ProtoDataConverter.h>

#include <cassert>
#include <cstddef>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DT>
struct Broadcast
{
    static void apply(Handle<DT> *&res, const DT *mat, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DT>
void broadcast(Handle<DT> *&res, const DT *mat, DCTX(ctx))
{
    Broadcast<DT>::apply(res, mat, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

template<>
struct Broadcast<DenseMatrix<double>>
{
    static void apply(Handle<DenseMatrix<double>> *&res, const DenseMatrix<double> *mat, DCTX(ctx))
    {
        auto envVar = std::getenv("DISTRIBUTED_WORKERS");
        assert(envVar && "Environment variable has to be set");
        std::string workersStr(envVar);
        std::string delimiter(",");

        size_t pos;
        std::vector<std::string> workers;

        while ((pos = workersStr.find(delimiter)) != std::string::npos) {
            workers.push_back(workersStr.substr(0, pos));
            workersStr.erase(0, pos + delimiter.size());
        }
        workers.push_back(workersStr);


        // auto blockSize = DistributedData::BLOCK_SIZE;

        struct StoredInfo {            
            DistributedIndex *ix ;
            std::string workerAddr;
            std::shared_ptr<grpc::Channel> channel;
        };
        // Keep a vector for holding data about all workers
        using StoredInfoVector = std::vector<StoredInfo>;
        StoredInfoVector storedInfoLeftChild, storedInfoRightChild;

        DistributedCaller<std::vector<StoredInfo>, distributed::BroadcastedData, distributed::BroadcastedStored> caller;
        
        Handle<DenseMatrix<double>>::HandleMap map;   

        distributed::Matrix protoMat;
        ProtoDataConverter::convertToProto(mat, &protoMat);
        
        
        // Split worker vector in half. Each child worker recieves half of the workers to continue the broadcast call
        std::string leftChildAddr, rightChildAddr;
        distributed::BroadcastedData dataLeftChild, dataRightChild;
        
        leftChildAddr = workers[0];
        rightChildAddr = workers[workers.size() / 2];
        
        // Left child
        auto channelLeftChild = caller.GetOrCreateChannel(leftChildAddr);
        storedInfoLeftChild.push_back({new DistributedIndex(0, 0), leftChildAddr, channelLeftChild});
        
        
        // Children of left child
        for (auto i = 1ul; i < workers.size() / 2; i++){
            // Data sent to child worker
            // Protobuf's "repeated" fields retain the order of items
            // therefore we can ensure that DistributedIndexes are assigned as expected.
            dataLeftChild.add_addresses(workers[i]);
            // Store data
            auto channel = caller.GetOrCreateChannel(workers[i]);
            storedInfoLeftChild.push_back({new DistributedIndex(i, 0), workers[i], channel});
        }
        // Make call
        dataLeftChild.mutable_matrix()->CopyFrom(protoMat);
        caller.asyncBroadcastCall(leftChildAddr, storedInfoLeftChild, dataLeftChild);

        // Repeat for right child
        // Right Child is optional
        if (rightChildAddr != leftChildAddr){
            auto channelRightChild = caller.GetOrCreateChannel(rightChildAddr);
            storedInfoRightChild.push_back({new DistributedIndex(workers.size() / 2, 0), rightChildAddr, channelRightChild});

        
            // Children of right child
            for (auto i = (workers.size() / 2) + 1; i < workers.size(); i++){
                dataRightChild.add_addresses(workers[i]);
                
                auto channel = caller.GetOrCreateChannel(workers[i]);
                storedInfoRightChild.push_back({new DistributedIndex(i, 0), workers[i], channel});
            }
            dataRightChild.mutable_matrix()->CopyFrom(protoMat);
            caller.asyncBroadcastCall(rightChildAddr, storedInfoRightChild, dataRightChild);        
        }                    

        // get results
        while (!caller.isQueueEmpty()){
            auto response = caller.getNextResult();
            auto childStoredInfo = response.storedInfo;
            for (auto i = 0ul; i < childStoredInfo.size(); i++){
                auto item = childStoredInfo[i];                
                auto ix = item.ix;
                auto addr = item.workerAddr;
                auto channel = item.channel;
                // Get actuall results
                auto storedData = response.result.stored(i);
                DistributedData data(storedData, addr, channel);
                map.insert({*ix, data});
            }
        }
        res = new Handle<DenseMatrix<double>>(map, mat->getNumRows(), mat->getNumCols());
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_BROADCAST_H