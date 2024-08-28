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
#include <memory>

class AllocationDescriptorMPI : public IAllocationDescriptor {
    ALLOCATION_TYPE type = ALLOCATION_TYPE::DIST_MPI;
    int processRankID{};
    [[maybe_unused]] DaphneContext* ctx{};
    DistributedData distributedData;
    std::shared_ptr<std::byte> data;

public:
    AllocationDescriptorMPI() = default;
    AllocationDescriptorMPI(int id, DaphneContext* ctx, DistributedData& data) :  processRankID(id), ctx(ctx),
            distributedData(data) {}

    ~AllocationDescriptorMPI() override = default;

    [[nodiscard]] ALLOCATION_TYPE getType() const override { return type; };
    
    [[nodiscard]] std::string getLocation() const override { return std::to_string(processRankID); };

    [[nodiscard]] std::unique_ptr<IAllocationDescriptor>  createAllocation(size_t size, bool zero) const override {
        /* TODO */
        throw std::runtime_error("AllocationDescriptorGRPC::createAllocation not implemented");
    }

    std::shared_ptr<std::byte> getData() override {return nullptr;} ;

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

    [[maybe_unused]] [[nodiscard]] DistributedIndex getDistributedIndex() const { return distributedData.ix; }

    DistributedData getDistributedData() { return distributedData; }

    void updateDistributedData(DistributedData& data_) { distributedData = data_; }

    [[nodiscard]] int getRank() const { return processRankID; }
};
