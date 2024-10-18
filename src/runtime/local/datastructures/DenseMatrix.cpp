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

#include "DenseMatrix.h"
#include <runtime/local/datastructures/AllocationDescriptorHost.h>
#include <runtime/local/io/DaphneSerializer.h>

#include <fmt/core.h>
#include <spdlog/spdlog.h>

#include <sstream>
#include <stdexcept>

// ****************************************************************************
// Boundary Validation
// ****************************************************************************

template <typename ValueType>
void validateArgs(const DenseMatrix<ValueType> *src, int64_t rowLowerIncl, int64_t rowUpperExcl, int64_t colLowerIncl,
                  int64_t colUpperExcl) {
    if (src == nullptr)
        throw std::runtime_error("invalid argument passed to dense matrix "
                                 "constructor: src must not be null");

    if (rowLowerIncl < 0 || rowUpperExcl < rowLowerIncl || static_cast<ssize_t>(src->getNumRows()) < rowUpperExcl ||
        (rowLowerIncl == static_cast<ssize_t>(src->getNumRows()) && rowLowerIncl != 0)) {
        std::ostringstream errMsg;
        errMsg << "invalid arguments '" << rowLowerIncl << ", " << rowUpperExcl
               << "' passed to dense matrix constructor: it must hold 0 <= "
                  "rowLowerIncl <= rowUpperExcl <= #rows "
               << "and rowLowerIncl < #rows (unless both are zero) where #rows "
                  "of src is '"
               << src->getNumRows() << "'";
        throw std::out_of_range(errMsg.str());
    }

    if (colLowerIncl < 0 || colUpperExcl < colLowerIncl || static_cast<ssize_t>(src->getNumCols()) < colUpperExcl ||
        (colLowerIncl == static_cast<ssize_t>(src->getNumCols()) && colLowerIncl != 0)) {
        std::ostringstream errMsg;
        errMsg << "invalid arguments '" << colLowerIncl << ", " << colUpperExcl
               << "' passed to dense matrix constructor: it must hold 0 <= "
                  "colLowerIncl <= colUpperExcl <= #columns "
               << "and colLowerIncl < #columns (unless both are zero) where "
                  "#columns of src is '"
               << src->getNumCols() << "'";
        throw std::out_of_range(errMsg.str());
    }
}

// ****************************************************************************
//
// ****************************************************************************

template <typename ValueType>
DenseMatrix<ValueType>::DenseMatrix(size_t maxNumRows, size_t numCols, bool zero, IAllocationDescriptor *allocInfo)
    : Matrix<ValueType>(maxNumRows, numCols), is_view(false), rowSkip(numCols),
      bufferSize(numRows * numCols * sizeof(ValueType)), lastAppendedRowIdx(0), lastAppendedColIdx(0) {
    std::unique_ptr<IAllocationDescriptor> val_alloc;
    if (!allocInfo) {
        alloc_shared_values(zero);
        auto bytes = std::reinterpret_pointer_cast<std::byte>(values);
        val_alloc = AllocationDescriptorHost::createHostAllocation(bytes, getBufferSize(), zero);
    } else {
        val_alloc = allocInfo->createAllocation(getBufferSize(), zero);

        // ToDo: refactor data storage into memory management
        if (allocInfo->getType() == ALLOCATION_TYPE::HOST) {
            alloc_shared_values(zero);

            auto bytes = std::reinterpret_pointer_cast<std::byte>(values);
            dynamic_cast<AllocationDescriptorHost *>(val_alloc.get())->setData(bytes);
        }
    }
    spdlog::debug("Creating {} x {} dense matrix of type: {}. Required memory: {} Mb", numRows, numCols,
                  val_alloc->getTypeName(), static_cast<float>(getBufferSize()) / (1048576));

    std::vector<std::unique_ptr<IAllocationDescriptor>> allocations;
    allocations.emplace_back(std::move(val_alloc));
    auto p = this->mdo->addDataPlacement(allocations);
    this->mdo->addLatest(p->getID());
}

template <typename ValueType>
DenseMatrix<ValueType>::DenseMatrix(size_t numRows, size_t numCols, std::shared_ptr<ValueType[]> &values)
    : Matrix<ValueType>(numRows, numCols), is_view(false), rowSkip(numCols), values(values),
      bufferSize(numRows * numCols * sizeof(ValueType)), lastAppendedRowIdx(0), lastAppendedColIdx(0) {
    auto bytes = std::reinterpret_pointer_cast<std::byte>(values);

    std::vector<std::unique_ptr<IAllocationDescriptor>> allocations;
    auto vec_alloc = AllocationDescriptorHost::createHostAllocation(bytes, getBufferSize(), false);
    allocations.emplace_back(std::move(vec_alloc));

    auto p = this->mdo->addDataPlacement(allocations);
    this->mdo->addLatest(p->getID());
}

template <typename ValueType>
DenseMatrix<ValueType>::DenseMatrix(const DenseMatrix<ValueType> *src, int64_t rowLowerIncl, int64_t rowUpperExcl,
                                    int64_t colLowerIncl, int64_t colUpperExcl)
    : Matrix<ValueType>(rowUpperExcl - rowLowerIncl, colUpperExcl - colLowerIncl), is_view(true), rowSkip(src->rowSkip),
      bufferSize(numRows * numCols * sizeof(ValueType)), lastAppendedRowIdx(0), lastAppendedColIdx(0) {
    validateArgs(src, rowLowerIncl, rowUpperExcl, colLowerIncl, colUpperExcl);

    this->row_offset = isView() ? src->row_offset + rowLowerIncl : rowLowerIncl;
    this->col_offset = isView() ? src->col_offset + colLowerIncl : colLowerIncl;

    // ToDo: manage host mem (values) in a data placement
    if (src->values) {
        bufferSize = numRows * rowSkip * sizeof(ValueType);
        this->values = src->values;
    }
    this->mdo = src->mdo;
}

template <typename ValueType>
DenseMatrix<ValueType>::DenseMatrix(size_t numRows, size_t numCols, const DenseMatrix<ValueType> *src)
    : Matrix<ValueType>(numRows, numCols), is_view(src->is_view), rowSkip(src->getRowSkip()),
      bufferSize(numRows * numCols * sizeof(ValueType)), lastAppendedRowIdx(0), lastAppendedColIdx(0) {
    if (!is_view)
        rowSkip = numCols;

    this->row_offset = src->row_offset;
    this->col_offset = src->col_offset;
    this->values = src->values;
    this->mdo = src->mdo;
}

template <typename ValueType> void DenseMatrix<ValueType>::printValue(std::ostream &os, ValueType val) const {
    os << val;
}

// Convert to an integer to print uint8_t values as numbers
// even if they fall into the range of special ASCII characters.
template <> [[maybe_unused]] void DenseMatrix<uint8_t>::printValue(std::ostream &os, uint8_t val) const {
    os << static_cast<uint32_t>(val);
}

template <> [[maybe_unused]] void DenseMatrix<int8_t>::printValue(std::ostream &os, int8_t val) const {
    os << static_cast<int32_t>(val);
}

template <typename ValueType>
void DenseMatrix<ValueType>::alloc_shared_values(bool zero, std::shared_ptr<ValueType[]> src, size_t offset) {
    // correct since C++17: Calls delete[] instead of simple delete
    if (src) {
        values = std::shared_ptr<ValueType[]>(src, src.get() + offset);
    } else {
        values = std::shared_ptr<ValueType[]>(new ValueType[getBufferSize()]);
        if (zero)
            std::fill(values.get(), values.get() + this->getNumRows() * this->getNumCols(),
                      ValueTypeUtils::defaultValue<ValueType>);
    }
}

template <typename ValueType> bool DenseMatrix<ValueType>::operator==(const DenseMatrix<ValueType> &rhs) const {
    // Note that we do not use the generic `get` interface to matrices here since
    // this operator is meant to be used for writing tests for, besides others,
    // those generic interfaces.

    if (this == &rhs)
        return true;

    const size_t numRows = this->getNumRows();
    const size_t numCols = this->getNumCols();

    if (numRows != rhs.getNumRows() || numCols != rhs.getNumCols())
        return false;

    const ValueType *valuesLhs = this->getValues();
    const ValueType *valuesRhs = rhs.getValues();

    const size_t rowSkipLhs = this->getRowSkip();
    const size_t rowSkipRhs = rhs.getRowSkip();

    if (valuesLhs == valuesRhs && rowSkipLhs == rowSkipRhs)
        return true;

    for (size_t r = 0; r < numRows; ++r) {
        for (size_t c = 0; c < numCols; ++c) {
            if (*(valuesLhs + c) != *(valuesRhs + c))
                return false;
        }
        valuesLhs += rowSkipLhs;
        valuesRhs += rowSkipRhs;
    }
    return true;
}

template <typename ValueType> size_t DenseMatrix<ValueType>::serialize(std::vector<char> &buf) const {
    return DaphneSerializer<DenseMatrix<ValueType>>::serialize(this, buf);
}

template <> size_t DenseMatrix<bool>::serialize(std::vector<char> &buf) const {
    throw std::runtime_error("DenseMatrix<bool> serialization not implemented");
}

template <typename ValueType>
const ValueType *DenseMatrix<ValueType>::getValues(const IAllocationDescriptor *alloc_desc, const Range *range) const {
    const ValueType *ptr;
    if (this->isPinned(alloc_desc)) {
        ptr = reinterpret_cast<const ValueType *>(this->pinned_mem);
    } else {
        this->pinned_mem = this->mdo->getData(0, alloc_desc, range);
        ptr = reinterpret_cast<const ValueType *>(this->pinned_mem);
        const_cast<DenseMatrix<ValueType> *>(this)->pin(alloc_desc);
    }

    if (isView())
        return ptr + offset();
    else
        return ptr;
}

template <typename ValueType>
ValueType *DenseMatrix<ValueType>::getValues(const IAllocationDescriptor *alloc_desc, const Range *range) {
    ValueType *ptr;
    if (this->isPinned(alloc_desc)) {
        ptr = reinterpret_cast<ValueType *>(this->pinned_mem);
    } else {
        this->pinned_mem = this->mdo->getData(0, alloc_desc, range);
        ptr = reinterpret_cast<ValueType *>(this->pinned_mem);
        this->pin(alloc_desc);
    }

    if (isView())
        return ptr + offset();
    else
        return ptr;
}

// ----------------------------------------------------------------------------
// const char* specialization
DenseMatrix<const char *>::DenseMatrix(size_t maxNumRows, size_t numCols, bool zero, size_t strBufferCapacity_,
                                       ALLOCATION_TYPE type)
    : Matrix<const char *>(maxNumRows, numCols), rowSkip(numCols), lastAppendedRowIdx(0), lastAppendedColIdx(0) {
    spdlog::debug("creating dense matrix of allocation type {}, dims: {}x{} "
                  "req.mem.: {}Mb with at least {} bytes for strings",
                  static_cast<int>(type), numRows, numCols, printBufferSize(), strBufferCapacity_);
    if (type == ALLOCATION_TYPE::HOST) {
        alloc_shared_values();
        alloc_shared_strings(nullptr, strBufferCapacity_);
    } else {
        throw std::runtime_error("Unknown allocation type: " + std::to_string(static_cast<int>(type)));
    }
}

DenseMatrix<const char *>::DenseMatrix(size_t numRows, size_t numCols, std::shared_ptr<const char *[]> &strings_,
                                       size_t strBufferCapacity_, std::shared_ptr<const char *> cuda_ptr_)
    : Matrix<const char *>(numRows, numCols), rowSkip(numCols), cuda_ptr(cuda_ptr_), lastAppendedRowIdx(0),
      lastAppendedColIdx(0) {
    alloc_shared_values();
    alloc_shared_strings();
    prepareAppend();
    for (size_t r = 0; r < numRows; r++)
        for (size_t c = 0; c < numCols; c++)
            append(r, c, strings_.get()[r * rowSkip + c]);
    finishAppend();
}

DenseMatrix<const char *>::DenseMatrix(const DenseMatrix<const char *> *src, int64_t rowLowerIncl, int64_t rowUpperExcl,
                                       int64_t colLowerIncl, int64_t colUpperExcl)
    : Matrix<const char *>(rowUpperExcl - rowLowerIncl, colUpperExcl - colLowerIncl), lastAppendedRowIdx(0),
      lastAppendedColIdx(0) {
    validateArgs(src, rowLowerIncl, rowUpperExcl, colLowerIncl, colUpperExcl);

    rowSkip = src->rowSkip;
    auto offset = rowLowerIncl * src->rowSkip + colLowerIncl;
    alloc_shared_values(src->values, offset);
    alloc_shared_strings(src->strBuf);
}

void DenseMatrix<const char *>::printValue(std::ostream &os, const char *val) const { os << '\"' << val << "\""; }

void DenseMatrix<const char *>::alloc_shared_values(std::shared_ptr<const char *[]> src, size_t offset) {
    // correct since C++17: Calls delete[] instead of simple delete
    if (src) {
        values = std::shared_ptr<const char *[]>(src, src.get() + offset);
    } else {
        values = std::shared_ptr<const char *[]>(new const char *[getNumItems()]);
    }
}

void DenseMatrix<const char *>::alloc_shared_strings(std::shared_ptr<CharBuf> src, size_t strBufferCapacity_) {
    if (src) {
        strBuf = std::shared_ptr<CharBuf>(src);
    } else {
        if (!values)
            alloc_shared_values();
        strBuf = std::make_shared<CharBuf>(strBufferCapacity_, getNumItems());
        appendZerosRange(&values[0], getNumItems());
    }
}

// explicitly instantiate to satisfy linker
template class DenseMatrix<double>;
template class DenseMatrix<float>;
template class DenseMatrix<int>;
template class DenseMatrix<long>;
template class DenseMatrix<signed char>;
template class DenseMatrix<unsigned char>;
template class DenseMatrix<unsigned int>;
template class DenseMatrix<unsigned long>;
template class DenseMatrix<bool>;
template class DenseMatrix<std::string>;
template class DenseMatrix<FixedStr16>;
