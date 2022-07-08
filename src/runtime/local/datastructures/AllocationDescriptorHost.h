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

#include "ObjectMetaData.h"
#include <cstdint>

class AllocationDescriptorHost : public IAllocationDescriptor {
public:
    ~AllocationDescriptorHost() override = default;
    [[nodiscard]] ALLOCATION_TYPE getType() const override { return ALLOCATION_TYPE::HOST; }
    void createAllocation(size_t size, bool zero) override { }
    std::shared_ptr<std::byte> getData() override { return nullptr; }
    void transferTo(std::byte* src, size_t size) override { }
    void transferFrom(std::byte* dst, size_t size) override {}
    [[nodiscard]] std::unique_ptr<IAllocationDescriptor> clone() const override {
        return std::make_unique<AllocationDescriptorHost>(*this);
    }
    bool operator==(const IAllocationDescriptor* other) const override { return (getType() == other->getType()); }
};
