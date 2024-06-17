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

#include "DataPlacement.h"
#include <memory>
#include <utility>

class AllocationDescriptorHost : public IAllocationDescriptor {
    ALLOCATION_TYPE type = ALLOCATION_TYPE::HOST;
    std::shared_ptr<std::byte> data{};

  public:
    ~AllocationDescriptorHost() override = default;

    [[nodiscard]] ALLOCATION_TYPE getType() const override { return type; }

    static std::unique_ptr<AllocationDescriptorHost> createHostAllocation(std::shared_ptr<std::byte> data, size_t size,
                                                                          bool zero) {
        auto new_alloc = std::make_unique<AllocationDescriptorHost>();
        new_alloc->data = std::move(data);
        new_alloc->size = size;
        if (zero)
            memset(new_alloc->data.get(), 0, size);
        return new_alloc;
    }

    [[nodiscard]] std::unique_ptr<IAllocationDescriptor> createAllocation(size_t size, bool zero) const override {
        auto new_alloc = std::make_unique<AllocationDescriptorHost>();
        new_alloc->size = size;
        new_alloc->data = std::shared_ptr<std::byte>(new std::byte[size], std::default_delete<std::byte[]>());

        if (zero)
            memset(new_alloc->data.get(), 0, size);

        return new_alloc;
    }

    [[nodiscard]] std::string getLocation() const override { return "Host"; }
    std::shared_ptr<std::byte> getData() override { return data; }

    void setData(std::shared_ptr<std::byte> &_data) { data = _data; }

    void transferTo(std::byte *src, size_t size) override {}
    void transferFrom(std::byte *dst, size_t size) override {}
    [[nodiscard]] std::unique_ptr<IAllocationDescriptor> clone() const override {
        return std::make_unique<AllocationDescriptorHost>(*this);
    }

    bool operator==(const IAllocationDescriptor *other) const override { return (getType() == other->getType()); }
};
