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
    ObjectMetaData* omd;
    if(allocInfo != nullptr) {
#ifndef NDEBUG
        std::cout << "creating dense matrix of allocation type " << static_cast<int>(allocInfo->getType()) << ", dims: "
                << numRows << "x" << numCols << " req.mem.: " << static_cast<float>(bufferSize()) / (1048576) << "Mb"
                <<  std::endl;
#endif
        omd = this->addObjectMetaData(allocInfo);
        omd->allocation->createAllocation(bufferSize(), zero);
    }
    else {
        AllocationDescriptorHost myHostAllocInfo;
        alloc_shared_values();
        if(zero)
            memset(values.get(), 0, maxNumRows * numCols * sizeof(ValueType));
        omd = this->addObjectMetaData(&myHostAllocInfo);
    }
    this->omd_id = omd->omd_id;
    this->latest_version.push_back(this->omd_id);
}

template<typename ValueType>
DenseMatrix<ValueType>::DenseMatrix(const DenseMatrix * src, size_t rowLowerIncl, size_t rowUpperExcl, size_t colLowerIncl,
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
    ObjectMetaData* omd = this->addObjectMetaData(&myHostAllocInfo);
    this->omd_id = omd->omd_id;

}

template<typename ValueType>
const ValueType* DenseMatrix<ValueType>::getValuesInternal(const IAllocationDescriptor* alloc_desc, const Range* range) const {
    if(alloc_desc) {
        auto ret = this->findObjectMetaData(alloc_desc, range);
        if (!ret) {
            // alloc + transfer + add to latest
            // a bit of un-const-ness needed here :-/
            auto omd = const_cast<DenseMatrix<ValueType>*>(this)->addObjectMetaData(alloc_desc);
            omd->allocation->createAllocation(bufferSize(), false);
            omd->allocation->transferTo(reinterpret_cast<std::byte*>(values.get()), bufferSize());
            const_cast<DenseMatrix<ValueType>*>(this)->latest_version.push_back(omd->omd_id);
            return reinterpret_cast<ValueType*>(omd->allocation->getData().get());
        }
        else {
            // if not in latest, transfer + add to latest
            if(!this->isLatestVersion(ret->omd_id)) {
                ret->allocation->transferTo(reinterpret_cast<std::byte*>(values.get()), bufferSize());
                const_cast<DenseMatrix<ValueType>*>(this)->latest_version.push_back(ret->omd_id);
            }
            return reinterpret_cast<ValueType*>(ret->allocation->getData().get());
        }
    }
    else {
        AllocationDescriptorHost myHostAllocInfo;
        if(!values)
            const_cast<DenseMatrix *>(this)->alloc_shared_values();
        if(!this->findObjectMetaData(&myHostAllocInfo, nullptr)) {
            auto ret = this->getObjectMetaDataByID(this->omd_id);
            if(ret) {
                ret->allocation->transferFrom(reinterpret_cast<std::byte *>(values.get()), bufferSize());
                const_cast<DenseMatrix<ValueType> *>(this)->latest_version.push_back(this->omd_id);
            }
            else
                throw std::runtime_error("Error: no object meta data in matrix");
        }
        return values.get();
    }
}

template <typename ValueType> void DenseMatrix<ValueType>::printValue(std::ostream & os, ValueType val) const {
    os << val;
}

// Convert to an integer to print uint8_t values as numbers
// even if they fall into the range of special ASCII characters.
template <>
[[maybe_unused]] void DenseMatrix<unsigned char>::printValue(std::ostream & os, unsigned char val) const {
    os << static_cast<unsigned int>(val);
}

template <>
[[maybe_unused]] void DenseMatrix<signed char>::printValue(std::ostream & os, signed char val) const {
    os << static_cast<int>(val);
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

template<typename ValueType>
DenseMatrix<ValueType>* DenseMatrix<ValueType>::vectorTranspose() const {
    assert((this->numRows == 1 || this->numCols == 1) && "no-op transpose for vectors only");
    auto transposed = DataObjectFactory::create<DenseMatrix<ValueType>>(this, 0, this->getNumRows(), 0, this->getNumCols());
    std::swap(transposed->numRows, transposed->numCols);
    return transposed;
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
