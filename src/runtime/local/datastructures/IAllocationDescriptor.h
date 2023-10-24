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

#include <memory>

// An alphabetically sorted wishlist of supported allocation types ;-)
// Supporting all of that is probably unmaintainable :-/
enum class ALLOCATION_TYPE {
    DIST_GRPC, // Generic gRPC TAG
    DIST_GRPC_ASYNC, // Asynchronous gRPC communication
    DIST_GRPC_SYNC, // Synchronous gRPC communication
    DIST_MPI,
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

/**
 * @brief The IAllocationDescriptor interface class describes an abstract interface to handle memory allocations
 *
 * To decouple specifics of a certain API for managing memory (e.g, hardware accelerators, distributed libraries)
 * the allocation descriptor interface provides a set of methods that need to be implemented by the concrete API
 * specific derivations.
 * These allocation descriptors are used to request a certain type of memory when using the getValues() method of
 * a matrix/frame. They are also responsible for transferring to and from the special memory that is handled by
 * the allocator.
 */
class IAllocationDescriptor {
public:
    virtual ~IAllocationDescriptor() = default;
    [[nodiscard]] virtual ALLOCATION_TYPE getType() const = 0;
    virtual void createAllocation(size_t size, bool zero) = 0;
    virtual std::string getLocation() const = 0;
    virtual std::shared_ptr<std::byte> getData() = 0;
    virtual void transferTo(std::byte* src, size_t size) = 0;
    virtual void transferFrom(std::byte* dst, size_t size) = 0;
    [[nodiscard]] virtual std::unique_ptr<IAllocationDescriptor> clone() const = 0;
    virtual bool operator==(const IAllocationDescriptor* other) const { return (getType() == other->getType()); }
};
