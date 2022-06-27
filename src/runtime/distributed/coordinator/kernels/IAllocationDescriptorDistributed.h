/*
 * Copyright 2022 The DAPHNE Consortium
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

#pragma once

#include <runtime/local/datastructures/Structure.h>
#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/ObjectMetaData.h>
#include <ir/daphneir/Daphne.h>

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


struct DistributedData
{
    std::string filename;
    size_t numRows, numCols;
    mlir::daphne::VectorCombine vectorCombine;
    bool isPlacedAtWorker = false;
    DistributedIndex ix;

};

class IAllocationDescriptorDistributed : public IAllocationDescriptor {
protected: 
    DaphneContext* dctx{};
    std::string workerAddress;
    DistributedData data_;
public:
    IAllocationDescriptorDistributed() {} ;

    IAllocationDescriptorDistributed(DaphneContext* ctx, 
            std::string workerAddress) : 
            dctx(ctx), workerAddress(workerAddress) { }
    IAllocationDescriptorDistributed(DaphneContext* ctx, 
                            std::string workerAddress, 
                            DistributedData data) : 
                 dctx(ctx), workerAddress(workerAddress), data_(data) { }
    virtual ~IAllocationDescriptorDistributed() = default;    

    // Generic overriden implementations
    void createAllocation(size_t size, bool zero) override { std::runtime_error("not implemented"); }
    std::shared_ptr<std::byte> getData() override { std::runtime_error("not implemented"); }

    std::string getLocation() const override { return workerAddress; }

    // Primitives

    // Information that workers store and we need to retrieve after communication happens
    struct StoredInfo {
        std::string filename;
        size_t numRows, numCols;
    };
    // Distributed operations should return a vector, one entry for each result (multiple in case of compute primitive).
    // Each entry is a map with ObjectMetaData_ID as key
    // and Stored Info struct as value
    using DistributedResult = std::vector<std::map<size_t, StoredInfo>>;

    virtual DistributedResult Distribute(const Structure *arg) = 0;
    virtual DistributedResult Broadcast(const Structure *arg) = 0;
    // Broadcast a double. For now we also pass a Structure object which needs to be used for obj meta data.
    virtual DistributedResult Broadcast(const double *arg, const Structure *mat) = 0;
    
    // Compute cannot use key and ObjectMetadata_ID, 
    // it needs to map results using worker identifier (rank, address, etc)
    using DistributedComputeResult = std::vector<std::map<std::string, StoredInfo>>;
    virtual DistributedComputeResult Compute(const Structure **args, size_t numInputs, const char *mlirCode) = 0;
    virtual void Collect(Structure *arg) = 0; 

    const DistributedIndex getDistributedIndex()
    { return data_.ix; }    
    const DistributedData getDistributedData()
    { return data_; }
    void updateDistributedData(DistributedData data)
    { data_ = data; }
};
