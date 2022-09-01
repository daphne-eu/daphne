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
    std::string identifier;
    size_t numRows, numCols;
    mlir::daphne::VectorCombine vectorCombine;
    bool isPlacedAtWorker = false;
    DistributedIndex ix;

};

class AllocationDescriptorGRPC : public IAllocationDescriptor {
private:
    DaphneContext *ctx;
    ALLOCATION_TYPE type = ALLOCATION_TYPE::DIST_GRPC;
    std::string workerAddress;
    DistributedData data;
public:
    AllocationDescriptorGRPC() {} ;
    AllocationDescriptorGRPC(DaphneContext* ctx, 
                            std::string address, 
                            DistributedData data) : ctx(ctx), workerAddress(address), data(data) { } ;

    ~AllocationDescriptorGRPC() override {};
    [[nodiscard]] ALLOCATION_TYPE getType() const override 
    { return type; };
    
    std::string getLocation() const override 
    {return workerAddress; };
    void createAllocation(size_t size, bool zero) override {} ;
    std::shared_ptr<std::byte> getData() override {} ;

    bool operator==(const IAllocationDescriptor* other) const override {
        if(getType() == other->getType())
            return(getLocation() == dynamic_cast<const AllocationDescriptorGRPC *>(other)->getLocation());
        return false;
    } ;

    [[nodiscard]] std::unique_ptr<IAllocationDescriptor> clone() const override {
        return std::make_unique<AllocationDescriptorGRPC>(*this);
    }
    

    void transferTo(std::byte *src, size_t size) override { /* TODO */ };
    void transferFrom(std::byte *src, size_t size) override { /* TODO */ };

    const DistributedIndex getDistributedIndex()
    { return data.ix; }    
    const DistributedData getDistributedData()
    { return data; }
    void updateDistributedData(DistributedData data_)
    { data = data_; }
};
