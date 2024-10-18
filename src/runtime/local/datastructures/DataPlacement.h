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

#include "IAllocationDescriptor.h"
#include "Range.h"
#include <runtime/local/context/DaphneContext.h>

#include <atomic>

/**
 * The DataPlacement struct binds allocation descriptors to a range
 * description and stores an ID of an instantiated object.
 *
 * The number of allocations varys depending on the data type (1 allocation of values of a dense matrix
 * three allocations for values/col_idxs/row_ptrs of a CSR matrix)
 */
class DataPlacement {
    size_t dp_id;

    // used to generate object IDs
    static std::atomic_size_t instance_count;

    std::vector<std::unique_ptr<IAllocationDescriptor>> allocations{};

    std::unique_ptr<Range> range{};

  public:
    DataPlacement() = delete;
    DataPlacement(std::vector<std::unique_ptr<IAllocationDescriptor>> _a, std::unique_ptr<Range> _r)
        : dp_id(instance_count++), allocations(std::move(_a)), range(std::move(_r)) {}

    [[nodiscard]] size_t getID() const { return dp_id; }
    std::string getLocation() { return allocations.front()->getLocation(); }
    Range *getRange() { return range.get(); }
    void setRange(Range *r) { range.reset(r); }
    void setRange(std::unique_ptr<Range> r) { range = std::move(r); }
    uint32_t getNumAllocations() { return allocations.size(); }
    IAllocationDescriptor *getAllocation(uint32_t num) { return allocations[num].get(); }
};
