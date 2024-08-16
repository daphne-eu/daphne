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

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/Structure.h>
#include <ir/daphneir/Daphne.h>

#include <runtime/local/datastructures/DistributedAllocationHelpers.h>

class AllocationDescriptorGRPC : public IAllocationDescriptor {
private:
    DaphneContext *ctx;
    ALLOCATION_TYPE type = ALLOCATION_TYPE::DIST_GRPC;
    const std::string workerAddress;
    DistributedData distributedData;
    std::shared_ptr<std::byte> data;
public:
    AllocationDescriptorGRPC() {} ;
    AllocationDescriptorGRPC(DaphneContext* ctx, 
                            const std::string &address, 
                            const DistributedData &data) : ctx(ctx), workerAddress(address), distributedData(data) { } ;

    ~AllocationDescriptorGRPC() override {};
    [[nodiscard]] ALLOCATION_TYPE getType() const override 
    { return type; };
    
    std::string getLocation() const override 
    {return workerAddress; };
    void createAllocation(size_t size, bool zero) override {}
    // TODO: Implement transferTo and transferFrom functions
    std::shared_ptr<std::byte> getData() override { 
        throw std::runtime_error("TransferTo/From functions are not implemented yet.");
    }

    bool operator==(const IAllocationDescriptor* other) const override {
        if(getType() == other->getType())
            return(getLocation() == dynamic_cast<const AllocationDescriptorGRPC *>(other)->getLocation());
        return false;
    }

    [[nodiscard]] std::unique_ptr<IAllocationDescriptor> clone() const override {
        return std::make_unique<AllocationDescriptorGRPC>(*this);
    }
    
    /* 
    TODO:
    We currently do not support transferTo/From functions for gRPC.
    All communication is handled by the distributed kernels (e.g. Distribute.h).
    In order to support these functions we need to know what data-type we
    are sending (representation, value type, etc.) since the worker 
    needs to store with the appropriate representation. 
    This might not be necessary when issue #103 "(De)Serialization of data objects" is implemented.
    For now support for these is not necessary but we need to think about this...
    */
    void transferTo(std::byte *src, size_t size) override { 
        /* TODO */ 
        throw std::runtime_error("TransferTo (gRPC) function is not implemented yet.");
    };
    void transferFrom(std::byte *src, size_t size) override { 
        /* TODO */ 
        throw std::runtime_error("TransferFrom (gRPC) function is not implemented yet.");
    };

    const DistributedIndex getDistributedIndex()
    { return distributedData.ix; }    
    const DistributedData getDistributedData()
    { return distributedData; }
    void updateDistributedData(DistributedData data_)
    { distributedData = data_; }
};
