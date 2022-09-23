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
#include "runtime/local/context/CUDAContext.h"

class AllocationDescriptorCUDA : public IAllocationDescriptor {
    ALLOCATION_TYPE type = ALLOCATION_TYPE::GPU_CUDA;
    uint32_t device_id{};
    DaphneContext* dctx{};
    std::shared_ptr<std::byte> data{};
    size_t alloc_id{};

public:
    AllocationDescriptorCUDA() = delete;

    AllocationDescriptorCUDA(DaphneContext* ctx, uint32_t device_id) : device_id(device_id), dctx(ctx) { }

    ~AllocationDescriptorCUDA() override {
        // ToDo: for now we free if this is the last context-external ref to the buffer
        if(data.use_count() == 2) {
            CUDAContext::get(dctx, device_id)->free(alloc_id);
        }
    }

    [[nodiscard]] ALLOCATION_TYPE getType() const override { return type; }

    // [[nodiscard]] uint32_t getLocation() const { return device_id; }
    [[nodiscard]] std::string getLocation() const override { return std::to_string(device_id); }

    void createAllocation(size_t size, bool zero) override {
        auto ctx = CUDAContext::get(dctx, device_id);
        data = ctx->malloc(size, zero, alloc_id);
    }

    std::shared_ptr<std::byte> getData() override { return data; }

    [[nodiscard]] std::unique_ptr<IAllocationDescriptor> clone() const override {
        return std::make_unique<AllocationDescriptorCUDA>(*this);
    }

    void transferTo(std::byte* src, size_t size) override {
        CHECK_CUDART(cudaMemcpy(data.get(), src, size, cudaMemcpyHostToDevice));
    }
    void transferFrom(std::byte* dst, size_t size) override {
        CHECK_CUDART(cudaMemcpy(dst, data.get(), size, cudaMemcpyDeviceToHost));
    };

    bool operator==(const IAllocationDescriptor* other) const override {
        if(getType() == other->getType())
            return(getLocation() == dynamic_cast<const AllocationDescriptorCUDA *>(other)->getLocation());
        return false;
    }
};
