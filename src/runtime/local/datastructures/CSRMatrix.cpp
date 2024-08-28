/*
 * Copyright 2021 The DAPHNE Consortium
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

#include <runtime/local/io/DaphneSerializer.h>

#include "CSRMatrix.h"

template<typename ValueType>
CSRMatrix<ValueType>::CSRMatrix(size_t maxNumRows, size_t numCols, size_t maxNumNonZeros, bool zero, IAllocationDescriptor* allocInfo) :
        Matrix<ValueType>(maxNumRows, numCols), numRowsAllocated(maxNumRows), isRowAllocatedBefore(false),
        is_view(false), maxNumNonZeros(maxNumNonZeros), lastAppendedRowIdx(0) {
    auto val_buf_size = maxNumNonZeros * sizeof(ValueType);
    auto cidx_buf_size = maxNumNonZeros * sizeof(size_t);
    auto rptr_buf_size = (numRows + 1) * sizeof(size_t);

    std::unique_ptr<IAllocationDescriptor> val_alloc;
    std::unique_ptr<IAllocationDescriptor> cidx_alloc;
    std::unique_ptr<IAllocationDescriptor> rptr_alloc;

    if(!allocInfo) {
        values = std::shared_ptr<ValueType>(new ValueType[maxNumNonZeros], std::default_delete<ValueType[]>());
        auto bytes = std::reinterpret_pointer_cast<std::byte>(values);
        val_alloc = AllocationDescriptorHost::createHostAllocation(bytes, val_buf_size, zero);
        colIdxs = std::shared_ptr<size_t>(new size_t[maxNumNonZeros], std::default_delete<size_t[]>());
        bytes = std::reinterpret_pointer_cast<std::byte>(colIdxs);
        cidx_alloc = AllocationDescriptorHost::createHostAllocation(bytes, cidx_buf_size, zero);
        rowOffsets = std::shared_ptr<size_t>(new size_t[numRows + 1], std::default_delete<size_t[]>());
        bytes = std::reinterpret_pointer_cast<std::byte>(rowOffsets);
        rptr_alloc = AllocationDescriptorHost::createHostAllocation(bytes, rptr_buf_size, zero);
    }
    else {
        val_alloc = allocInfo->createAllocation(val_buf_size, zero);
        cidx_alloc = allocInfo->createAllocation(cidx_buf_size, zero);
        rptr_alloc = allocInfo->createAllocation(rptr_buf_size, zero);

        // ToDo: refactor data storage into memory management
        if(allocInfo->getType() == ALLOCATION_TYPE::HOST) {
            values = std::reinterpret_pointer_cast<ValueType>(val_alloc->getData());
            colIdxs = std::reinterpret_pointer_cast<size_t>(cidx_alloc->getData());
            rowOffsets = std::reinterpret_pointer_cast<size_t>(rptr_alloc->getData());
        }
    }

    spdlog::debug("Creating {} x {} sparse matrix of type: {}. Required memory: vals={}, cidxs={}, rptrs={}, total={} Mb", numRows, numCols,
                  val_alloc->getTypeName(), static_cast<float>(val_buf_size) / (1048576), static_cast<float>(cidx_buf_size) / (1048576),
                  static_cast<float>(rptr_buf_size) / (1048576), static_cast<float>(val_buf_size + cidx_buf_size + rptr_buf_size) / (1048576));

    std::vector<std::unique_ptr<IAllocationDescriptor>> allocations;
    allocations.emplace_back(std::move(val_alloc));
    allocations.emplace_back(std::move(cidx_alloc));
    allocations.emplace_back(std::move(rptr_alloc));
    auto p = this->mdo->addDataPlacement(allocations);
    this->mdo->addLatest(p->getID());
}

template<typename ValueType>
size_t CSRMatrix<ValueType>::serialize(std::vector<char> &buf) const {
    return DaphneSerializer<CSRMatrix<ValueType>>::serialize(this, buf);
}

// explicitly instantiate to satisfy linker
template class CSRMatrix<double>;
template class CSRMatrix<float>;
template class CSRMatrix<int>;
template class CSRMatrix<long>;
template class CSRMatrix<signed char>;
template class CSRMatrix<unsigned char>;
template class CSRMatrix<unsigned int>;
template class CSRMatrix<unsigned long>;
