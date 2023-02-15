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
template<typename ValueType>
DenseMatrix<ValueType>::DenseMatrix(size_t maxNumRows, size_t numCols, bool zero, IAllocationDescriptor* allocInfo) :
        Matrix<ValueType>(maxNumRows, numCols), rowSkip(numCols), lastAppendedRowIdx(0), lastAppendedColIdx(0)
{
    DataPlacement* new_data_placement;
    if(allocInfo != nullptr) {
#ifndef NDEBUG
        std::cerr << "creating dense matrix of allocation type " << static_cast<int>(allocInfo->getType()) << ", dims: "
                << numRows << "x" << numCols << " req.mem.: " << static_cast<float>(bufferSize()) / (1048576) << "Mb"
                <<  std::endl;
#endif
        new_data_placement = this->mdo.addDataPlacement(allocInfo);
        new_data_placement->allocation->createAllocation(bufferSize(), zero);
    }
    else {
        AllocationDescriptorHost myHostAllocInfo;
        alloc_shared_values();
        if(zero)
            memset(values.get(), 0, maxNumRows * numCols * sizeof(ValueType));
        new_data_placement = this->mdo.addDataPlacement(&myHostAllocInfo);
    }
    this->mdo.addLatest(new_data_placement->dp_id);
}

template<typename ValueType>
DenseMatrix<ValueType>::DenseMatrix(size_t numRows, size_t numCols, std::shared_ptr<ValueType[]>& values) :
        Matrix<ValueType>(numRows, numCols), rowSkip(numCols), values(values), lastAppendedRowIdx(0),
        lastAppendedColIdx(0) {
    AllocationDescriptorHost myHostAllocInfo;
    DataPlacement* new_data_placement = this->mdo.addDataPlacement(&myHostAllocInfo);
    this->mdo.addLatest(new_data_placement->dp_id);
}

template<typename ValueType>
DenseMatrix<ValueType>::DenseMatrix(ValueType *memRefPtr) : Matrix<ValueType>(10, 10){
    this->memRefPtr = memRefPtr;
    std::cout << "creating DM with 10x10 and ptr@: " << memRefPtr << " \n";
}

template<typename ValueType>
DenseMatrix<ValueType>::DenseMatrix(const DenseMatrix<ValueType> * src, size_t rowLowerIncl, size_t rowUpperExcl, size_t colLowerIncl,
        size_t colUpperExcl) : Matrix<ValueType>(rowUpperExcl - rowLowerIncl, colUpperExcl - colLowerIncl),
        lastAppendedRowIdx(0), lastAppendedColIdx(0)
{
    assert(src && "src must not be null");
    assert((rowLowerIncl < src->numRows) && "rowLowerIncl is out of bounds");
    assert((rowUpperExcl <= src->numRows) && "rowUpperExcl is out of bounds");
    assert((rowLowerIncl < rowUpperExcl) && "rowLowerIncl must be lower than rowUpperExcl");
    assert((colLowerIncl < src->numCols) && "colLowerIncl is out of bounds");
    assert((colUpperExcl <= src->numCols) && "colUpperExcl is out of bounds");
    assert((colLowerIncl < colUpperExcl) && "colLowerIncl must be lower than colUpperExcl");

    rowSkip = src->rowSkip;
    auto offset = rowLowerIncl * src->rowSkip + colLowerIncl;
    alloc_shared_values(src->values, offset);
    // ToDo: handle object meta data
    AllocationDescriptorHost myHostAllocInfo;
    auto new_placement = this->mdo.addDataPlacement(&myHostAllocInfo);
    this->mdo.addLatest(new_placement->dp_id);
}

template<typename ValueType>
auto DenseMatrix<ValueType>::getValuesInternal(const IAllocationDescriptor* alloc_desc, const Range* range)
        -> std::tuple<bool, size_t, ValueType*> {
    // If no range information is provided we assume the full range that this matrix covers
    if(range == nullptr || *range == Range(*this)) {
        if(alloc_desc) {
            auto ret = this->mdo.findDataPlacementByType(alloc_desc, range);
            if(!ret) {
                // find other allocation type X (preferably host allocation) to transfer from in latest_version

                // tuple content: <is latest, latest-id, ptr-to-data-placement>
                std::tuple<bool, size_t, ValueType *> result = std::make_tuple(false, 0, nullptr);
                auto latest = this->mdo.getLatest();
                DataPlacement *placement;
                for (auto &placement_id: latest) {
                    placement = this->mdo.getDataPlacementByID(placement_id);
                    if(placement->range == nullptr || *(placement->range) == Range{0, 0, this->getNumRows(),
                                                                                   this->getNumCols()}) {
                        std::get<0>(result) = true;
                        std::get<1>(result) = placement->dp_id;
                        // prefer host allocation
                        if(placement->allocation->getType() == ALLOCATION_TYPE::HOST) {
                            std::get<2>(result) = reinterpret_cast<ValueType *>(values.get());
                            break;
                        }
                    }
                }

                // if we found a data placement that is not in host memory, transfer it there before returning
                if(std::get<0>(result) == true && std::get<2>(result) == nullptr) {
                    AllocationDescriptorHost myHostAllocInfo;
                    if(!values)
                        this->alloc_shared_values();
                    this->mdo.addDataPlacement(&myHostAllocInfo);
                    placement->allocation->transferFrom(reinterpret_cast<std::byte *>(values.get()), bufferSize());
                    std::get<2>(result) = values.get();
                }

                // create new data placement
                auto new_data_placement = const_cast<DenseMatrix<ValueType> *>(this)->mdo.addDataPlacement(alloc_desc);
                new_data_placement->allocation->createAllocation(bufferSize(), false);

                // transfer to requested data placement
                new_data_placement->allocation->transferTo(reinterpret_cast<std::byte *>(values.get()), bufferSize());
                return std::make_tuple(false, new_data_placement->dp_id, reinterpret_cast<ValueType *>(
                        new_data_placement->allocation->getData().get()));
            }
            else {
                bool latest = this->mdo.isLatestVersion(ret->dp_id);
                if(!latest) {
                    ret->allocation->transferTo(reinterpret_cast<std::byte *>(values.get()), bufferSize());
                }
                return std::make_tuple(latest, ret->dp_id, reinterpret_cast<ValueType *>(ret->allocation->getData()
                        .get()));
            }
        }
        else {
            // if no alloc info was provided we try to get/create a full host allocation and return that
            std::tuple<bool, size_t, ValueType *> result = std::make_tuple(false, 0, nullptr);
            auto latest = this->mdo.getLatest();
            DataPlacement *placement;
            for (auto &placement_id: latest) {
                placement = this->mdo.getDataPlacementByID(placement_id);
                if(placement->range == nullptr || *(placement->range) == Range{0, 0, this->getNumRows(),
                        this->getNumCols()}) {
                    std::get<0>(result) = true;
                    std::get<1>(result) = placement->dp_id;
                    // prefer host allocation
                    if(placement->allocation->getType() == ALLOCATION_TYPE::HOST) {
                        std::get<2>(result) = reinterpret_cast<ValueType *>(values.get());
                        break;
                    }
                }
            }

            // if we found a data placement that is not in host memory, transfer it there before returning
            if(std::get<0>(result) == true && std::get<2>(result) == nullptr) {
                AllocationDescriptorHost myHostAllocInfo;
                if(!values)
                    const_cast<DenseMatrix<ValueType> *>(this)->alloc_shared_values();
                this->mdo.addDataPlacement(&myHostAllocInfo);
                placement->allocation->transferFrom(reinterpret_cast<std::byte *>(values.get()), bufferSize());
                std::get<2>(result) = values.get();
            }
            if(std::get<2>(result) == nullptr)
                throw std::runtime_error("Error: no object meta data in matrix");
            else
                return result;
        }
    }
    else
        throw std::runtime_error("Error: range support under construction");
}

template <typename ValueType> void DenseMatrix<ValueType>::printValue(std::ostream & os, ValueType val) const {
    os << val;
}

// Convert to an integer to print uint8_t values as numbers
// even if they fall into the range of special ASCII characters.
template <>
[[maybe_unused]] void DenseMatrix<uint8_t>::printValue(std::ostream & os, uint8_t val) const {
    os << static_cast<uint32_t>(val);
}

template <>
[[maybe_unused]] void DenseMatrix<int8_t>::printValue(std::ostream & os, int8_t val) const {
    os << static_cast<int32_t>(val);
}

template<typename ValueType>
void DenseMatrix<ValueType>::alloc_shared_values(std::shared_ptr<ValueType[]> src, size_t offset) {
    // correct since C++17: Calls delete[] instead of simple delete
    if(src) {
        values = std::shared_ptr<ValueType[]>(src, src.get() + offset);
    }
    else
        values = std::shared_ptr<ValueType[]>(new ValueType[numRows*numCols]);
}

template<typename ValueType>
size_t DenseMatrix<ValueType>::bufferSize() {
    return this->getNumItems() * sizeof(ValueType);
}





// ----------------------------------------------------------------------------
// const char* specialization
DenseMatrix<const char*>::DenseMatrix(size_t maxNumRows, size_t numCols, bool zero, size_t strBufferCapacity_, ALLOCATION_TYPE type) :
        Matrix<const char*>(maxNumRows, numCols), rowSkip(numCols), lastAppendedRowIdx(0), lastAppendedColIdx(0)
{
#ifndef NDEBUG
    std::cerr << "creating dense matrix of allocation type " << static_cast<int>(type) <<
              ", dims: " << numRows << "x" << numCols << " req.mem.: " << printBufferSize() << "Mb" << " with at least " << strBufferCapacity_ << " bytes for strings" <<  std::endl;
#endif
    if (type == ALLOCATION_TYPE::HOST) {
        alloc_shared_values();
        alloc_shared_strings(nullptr, strBufferCapacity_);
    }
    else {
        throw std::runtime_error("Unknown allocation type: " + std::to_string(static_cast<int>(type)));
    }
}

DenseMatrix<const char*>::DenseMatrix(size_t numRows, size_t numCols, std::shared_ptr<const char*[]>& strings_, size_t strBufferCapacity_, std::shared_ptr<const char*> cuda_ptr_):
            Matrix<const char*>(numRows, numCols), rowSkip(numCols), cuda_ptr(cuda_ptr_), lastAppendedRowIdx(0), lastAppendedColIdx(0){
                alloc_shared_values();
                alloc_shared_strings();
                prepareAppend();
                for(size_t r = 0; r < numRows; r++)
                    for(size_t c = 0; c < numCols; c++)
                        append(r, c, strings_.get()[r * rowSkip + c]);
                finishAppend();
            }

DenseMatrix<const char*>::DenseMatrix(const DenseMatrix<const char*> * src, size_t rowLowerIncl, size_t rowUpperExcl, size_t colLowerIncl,
        size_t colUpperExcl) : Matrix<const char*>(rowUpperExcl - rowLowerIncl, colUpperExcl - colLowerIncl), lastAppendedRowIdx(0), lastAppendedColIdx(0)
{
    assert(src && "src must not be null");
    assert((rowLowerIncl < src->numRows) && "rowLowerIncl is out of bounds");
    assert((rowUpperExcl <= src->numRows) && "rowUpperExcl is out of bounds");
    assert((rowLowerIncl < rowUpperExcl) && "rowLowerIncl must be lower than rowUpperExcl");
    assert((colLowerIncl < src->numCols) && "colLowerIncl is out of bounds");
    assert((colUpperExcl <= src->numCols) && "colUpperExcl is out of bounds");
    assert((colLowerIncl < colUpperExcl) && "colLowerIncl must be lower than colUpperExcl");

    rowSkip = src->rowSkip;
    auto offset = rowLowerIncl * src->rowSkip + colLowerIncl;
    alloc_shared_values(src->values, offset);
    alloc_shared_strings(src->strBuf);
}

void DenseMatrix<const char*>::printValue(std::ostream & os, const char* val) const {
    os << '\"' << val << "\"";
}

void DenseMatrix<const char*>::alloc_shared_values(std::shared_ptr<const char*[]> src, size_t offset) {
    // correct since C++17: Calls delete[] instead of simple delete
    if(src) {
        values = std::shared_ptr<const char*[]>(src, src.get() + offset);
    }
    else
    {
        values = std::shared_ptr<const char*[]>(new const char*[getNumItems()]);
    }
}

void DenseMatrix<const char*>::alloc_shared_strings(std::shared_ptr<CharBuf> src, size_t strBufferCapacity_) {
    if(src) {
        strBuf = std::shared_ptr<CharBuf>(src);
    }
    else
    {
        if(!values)
            alloc_shared_values();
        strBuf = std::make_shared<CharBuf>(strBufferCapacity_, getNumItems());
        appendZerosRange(&values[0], getNumItems());
    }
}

size_t DenseMatrix<const char*>::bufferSize() {
    return this->getNumItems() * sizeof(const char*);
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
template class DenseMatrix<const char*>;
