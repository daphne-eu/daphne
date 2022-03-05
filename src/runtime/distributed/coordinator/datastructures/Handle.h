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

#ifndef SRC_RUNTIME_DISTRIBUTED_COORDINATOR_DATASTRUCTURES_HANDLE_H
#define SRC_RUNTIME_DISTRIBUTED_COORDINATOR_DATASTRUCTURES_HANDLE_H

#include <vector>
#include <string>

#include <grpcpp/grpcpp.h>

#include <runtime/distributed/proto/worker.pb.h>
#include <runtime/distributed/proto/worker.grpc.pb.h>

#include <runtime/distributed/coordinator/kernels/DistributedCaller.h>

class DistributedIndex
{
public:
    DistributedIndex() : row_(0), col_(0)
    {}
    DistributedIndex(size_t row, size_t col) : row_(row), col_(col)
    {}

    size_t getRow() const
    {
        return row_;
    }
    size_t getCol() const
    {
        return col_;
    }

    bool operator<(const DistributedIndex rhs) const
    {
        if (row_ < rhs.row_)
            return true;
        else if (row_ == rhs.row_)
            return col_ < rhs.col_;
        return false;
    }

private:
    size_t row_;
    size_t col_;
};

class DistributedData
{
public:
    // TODO: make configurable
    const static size_t BLOCK_SIZE = 512;

    DistributedData(DistributedIndex ix, distributed::StoredData data)
        : ix_(ix), data_(data)
    {}
    DistributedData(distributed::StoredData data)
        : data_(data)
    {}

    const DistributedIndex getDistributedIndex() const
    { return ix_; }
    const distributed::StoredData getData() const
    { return data_; }
private:
    DistributedIndex ix_;
    distributed::StoredData data_;    
};


template<class DT>
class Handle
{
public:
    struct WorkerInfo {
        std::vector<DistributedData> distributedDataArray;
        std::shared_ptr<grpc::Channel> channel_;
    };
    // TODO change DistributedData (no need to hold workerAddress there anymore)
    using HandleMap_v2 = std::map<std::string, WorkerInfo>;

    Handle (std::vector<std::string> workerAddresses) {
        for (auto addr : workerAddresses){            
            // TODO Save created channel
            map_.insert({addr, {std::vector<DistributedData>(), nullptr}});
        }
    };
    
    ~Handle() 
    { 
        // TODO updated new destructor 
        // DistributedCaller<void*, distributed::StoredData, distributed::Empty> caller;
        // // Free memory on the workers
        // for (auto &pair : map_) {
        //     auto data = pair.second.getData();
        //     auto channel = pair.second.getChannel();
        //     caller.asyncFreeMemCall(channel, nullptr, data);
        // }
        // // Check workers' respond status        
        // while (!caller.isQueueEmpty()){
        //     // caller obj checks for status
        //     auto response = caller.getNextResult();            
        // }
    }

    const HandleMap_v2 getMap() const
    { return map_; }

    void insertData(std::string addr, DistributedData data){
        map_[addr].distributedDataArray.push_back(data);
    }

private:
    HandleMap_v2 map_;
};

#endif //SRC_RUNTIME_DISTRIBUTED_COORDINATOR_DATASTRUCTURES_HANDLE_H