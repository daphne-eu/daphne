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
#include <runtime/distributed/coordinator/kernels/IAllocationDescriptorDistributed.h>

class AllocationDescriptorDistributedGRPC : public IAllocationDescriptorDistributed {
private:
    ALLOCATION_TYPE type = ALLOCATION_TYPE::DIST_GRPC;
public:
    AllocationDescriptorDistributedGRPC() {} ;
    AllocationDescriptorDistributedGRPC(DaphneContext* ctx, 
                            std::string workerAddress, 
                            DistributedData data) : 
                 IAllocationDescriptorDistributed(ctx, workerAddress, data) { };

    [[nodiscard]] ALLOCATION_TYPE getType() const override 
    { return type; };
    
    [[nodiscard]] std::unique_ptr<IAllocationDescriptor> clone() const override {
        return std::make_unique<AllocationDescriptorDistributedGRPC>(*this);
    }
    

    void transferTo(std::byte *src, size_t size) override { /* TODO */ };
    void transferFrom(std::byte *src, size_t size) override { /* TODO */ };

    // Primitives
    DistributedResult Distribute(const Structure *arg) override;
    DistributedResult Broadcast(const Structure *arg) override;
    DistributedComputeResult Compute(const Structure **args, size_t numInputs, const char *mlirCode) override;
    void Collect(Structure *arg) override;
};
