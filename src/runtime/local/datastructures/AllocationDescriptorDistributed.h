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

#include <cstdint>
#include <string>
#include "ObjectMetaData.h"
#include "runtime/local/context/DaphneContext.h"

enum DISTRIBUTED_BACKEND {
    MPI,
    GRPC
};

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


class AllocationDescriptorDistributed : public IAllocationDescriptor {
    ALLOCATION_TYPE type = ALLOCATION_TYPE::DISTRIBUTED;
    DaphneContext* dctx{};
    std::string workerAddress;

    DistributedData data_;


public:    
    
    
    AllocationDescriptorDistributed() = delete;

    AllocationDescriptorDistributed(DaphneContext* ctx, 
            std::string workerAddress) : 
            dctx(ctx), workerAddress(workerAddress) { }
    AllocationDescriptorDistributed(DaphneContext* ctx, 
                            std::string workerAddress, 
                            DistributedData data) : 
                 dctx(ctx), workerAddress(workerAddress), data_(data) { }

    ~AllocationDescriptorDistributed() override {
        // missing for now
    }

    [[nodiscard]] ALLOCATION_TYPE getType() const override { return type; }

    std::string getLocation() const override { return workerAddress; }

    void createAllocation(size_t size, bool zero) override {
        
    }

    std::shared_ptr<std::byte> getData() override { return nullptr; }

    [[nodiscard]] std::unique_ptr<IAllocationDescriptor> clone() const override {
        return std::make_unique<AllocationDescriptorDistributed>(*this);
    }

    void transferTo(std::byte* src, size_t size) override {
        std::runtime_error("transferTo grpc not implemented");
    }
    void transferFrom(std::byte* dst, size_t size) override {
        std::runtime_error("transferFrom grpc not implemented");
    };

    bool operator==(const IAllocationDescriptor* other) const override {
        if(getType() == other->getType())
            return(getLocation() == dynamic_cast<const AllocationDescriptorDistributed *>(other)->getLocation());
        return false;
    }
    const DistributedIndex getDistributedIndex()
    { return data_.ix; }
    
    const DistributedData getDistributedData()
    { return data_; }
    
    void updateDistributedData(DistributedData data)
    { data_ = data; }
};

