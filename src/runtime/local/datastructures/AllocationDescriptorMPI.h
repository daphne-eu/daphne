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
#include "DataPlacement.h"
#include <string>

class AllocationDescriptorMPI : public IAllocationDescriptor {
    ALLOCATION_TYPE type = ALLOCATION_TYPE::DIST_MPI;
    int processRankID;
    DaphneContext* ctx;
    std::shared_ptr<std::byte> data{};

public:
    AllocationDescriptorMPI() = delete;
    AllocationDescriptorMPI(DaphneContext* ctx, 
                            std::string address, 
                            DistributedData data) : ctx(ctx), workerAddress(address), data(data) { } ;

    ~AllocationDescriptorMPI() override {};
    [nodiscard]] ALLOCATION_TYPE getType() const override 
    { return type; };
    
    std::string getLocation() const override {
        return to_string(processRankID); 
    };
    
    void createAllocation(size_t size, bool zero) override {} ;
    std::shared_ptr<std::byte> getData() override {} ;

    bool operator==(const IAllocationDescriptor* other) const override {
        if(getType() == other->getType())
            return(getLocation() == dynamic_cast<const AllocationDescriptorMPI *>(other)->getLocation());
        return false;
    } ;

    [[nodiscard]] std::unique_ptr<IAllocationDescriptor> clone() const override {
        return std::make_unique<AllocationDescriptorMPI>(*this);
    }
    
    void transferTo(std::byte *src, size_t size) override { /* TODO */ };
    void transferFrom(std::byte *src, size_t size) override { /* TODO */ };

    //const DistributedIndex getDistributedIndex()
    //{ return data.ix; }    
    //const DistributedData getDistributedData()
    //{ return data; }
    //void updateDistributedData(DistributedData data_)
    //{ data = data_; }
};
