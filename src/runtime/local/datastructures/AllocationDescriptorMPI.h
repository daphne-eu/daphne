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

#ifndef SRC_RUNTIME_LOCAL_DATASTRUCTURE_ALLOCATION_DESCRIPTORMPH_H
#define SRC_RUNTIME_LOCAL_DATASTRUCTURE_ALLOCATION_DESCRIPTORMPH_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/Structure.h>
#include <ir/daphneir/Daphne.h>

#include <runtime/local/datastructures/DistributedAllocationHelpers.h>
#include <memory>

class AllocationDescriptorMPI : public IAllocationDescriptor {
    ALLOCATION_TYPE type = ALLOCATION_TYPE::DIST_MPI;
    int processRankID;
    DaphneContext* ctx;
    DistributedData distributedData;
    std::shared_ptr<std::byte> data;

public:
    AllocationDescriptorMPI() {} ;
    AllocationDescriptorMPI(int id,
                            DaphneContext* ctx, 
                            DistributedData data) :  processRankID(id), ctx(ctx), distributedData(data) {} ;

    ~AllocationDescriptorMPI() override {};

    [[nodiscard]] ALLOCATION_TYPE getType() const override 
    { return type; };
    
    std::string getLocation() const override {
        return std::to_string(processRankID); 
    };
    
    void createAllocation(size_t size, bool zero) override {} ;
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

    const DistributedIndex getDistributedIndex()
    { return distributedData.ix; }    
    const DistributedData getDistributedData()
    { return distributedData; }
    void updateDistributedData(DistributedData data_)
    { distributedData = data_; }
    int getRank()
    { return processRankID; }
};

#endif //SRC_RUNTIME_LOCAL_DATASTRUCTURE_ALLOCATION_DESCRIPTORMPH_H
