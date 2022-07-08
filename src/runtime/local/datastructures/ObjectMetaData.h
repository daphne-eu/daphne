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

// An alphabetically sorted wishlist of supported allocation types ;-)
// Supporting all of that is probably unmaintainable :-/
enum class ALLOCATION_TYPE {
    DIST_GRPC,
    DIST_OPENMPI,
    DIST_SPARK,
    GPU_CUDA,
    GPU_HIP,
    HOST,
    HOST_PINNED_CUDA,
    FPGA_INT, // Intel
    FPGA_XLX, // Xilinx
    ONEAPI, // probably need separate ones for CPU/GPU/FPGA
    NUM_ALLOC_TYPES
};

// Unused for now. This can be used to track sub allocations of matrices
struct Range {
    size_t r_start;
    size_t c_start;
    size_t r_len;
    size_t c_len;
    bool operator==(const Range* other) const {
        return((other != nullptr) && (r_start == other->r_start && c_start == other->c_start && r_len == other->r_len &&
                c_len == other->c_len));
    }
    [[nodiscard]] std::unique_ptr<Range> clone() const { return std::make_unique<Range>(*this); }
};

class IAllocationDescriptor {
public:
    virtual ~IAllocationDescriptor() = default;
    [[nodiscard]] virtual ALLOCATION_TYPE getType() const = 0;
    virtual void createAllocation(size_t size, bool zero) = 0;
    virtual std::shared_ptr<std::byte> getData() = 0;
    virtual void transferTo(std::byte* src, size_t size) = 0;
    virtual void transferFrom(std::byte* dst, size_t size) = 0;
    [[nodiscard]] virtual std::unique_ptr<IAllocationDescriptor> clone() const = 0;
    virtual bool operator==(const IAllocationDescriptor* other) const { return (getType() == other->getType()); }
};

struct ObjectMetaData {
    size_t omd_id;

    // used to generate object IDs
    static size_t instance_count;
    
    std::unique_ptr<IAllocationDescriptor> allocation{};

    std::unique_ptr<Range> range{};

    ObjectMetaData() = delete;
    ObjectMetaData(std::unique_ptr<IAllocationDescriptor> _a, std::unique_ptr<Range> _r) : omd_id(instance_count++),
                                                                                           allocation(std::move(_a)), range(std::move(_r)) { }
};
