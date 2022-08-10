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

// TODO DenseMatrix should not be concerned about CUDA.

#include "DenseMatrix.h"
#include <chrono>

#ifdef USE_CUDA
    #include <runtime/local/kernels/CUDA/HostUtils.h>
#endif

template<typename ValueType>
DenseMatrix<ValueType>::DenseMatrix(size_t maxNumRows, size_t numCols, bool zero, ALLOCATION_TYPE type) :
        Matrix<ValueType>(maxNumRows, numCols), rowSkip(numCols), lastAppendedRowIdx(0), lastAppendedColIdx(0)
{
#ifndef NDEBUG
    std::cout << "creating dense matrix of allocation type " << static_cast<int>(type) <<
              ", dims: " << numRows << "x" << numCols << " req.mem.: " << printBufferSize() << "Mb" <<  std::endl;
#endif
    if (type == ALLOCATION_TYPE::HOST_ALLOC) {
        alloc_shared_values();
        host_buffer_current = true;
        if(zero)
            memset(values.get(), 0, maxNumRows * numCols * sizeof(ValueType));
    }
    else if (type == ALLOCATION_TYPE::CUDA_ALLOC) {
        alloc_shared_cuda_buffer();
        cuda_buffer_current = true;
    }
    else {
        throw std::runtime_error("Unknown allocation type: " + std::to_string(static_cast<int>(type)));
    }
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
    host_dirty = src->host_dirty;
    host_buffer_current = src->host_buffer_current;

#ifdef USE_CUDA
    cuda_dirty = src->cuda_dirty;
    cuda_buffer_current = src->cuda_buffer_current;

    if(src->cuda_ptr)
        alloc_shared_cuda_buffer(src->cuda_ptr, offset);
#endif
}

template <typename ValueType> void DenseMatrix<ValueType>::printValue(std::ostream & os, ValueType val) const {
    os << val;
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

    auto transposed = DataObjectFactory::create<DenseMatrix<ValueType>>(this->getNumCols(), this->getNumRows(),
                                                                        this->getValuesSharedPtr(), this->getCUDAValuesSharedPtr());
    transposed->cuda_dirty = this->cuda_dirty;
    transposed->cuda_buffer_current = this->cuda_buffer_current;
    transposed->host_dirty = this->host_dirty;
    transposed->host_buffer_current = this->host_buffer_current;
    return transposed;
}

DenseMatrix<const char*>::DenseMatrix(size_t maxNumRows, size_t numCols, bool zero, size_t strBufferCapacity_, ALLOCATION_TYPE type) :
        Matrix<const char*>(maxNumRows, numCols), rowSkip(numCols), lastAppendedRowIdx(0), lastAppendedColIdx(0)
{
#ifndef NDEBUG
    std::cout << "creating dense matrix of allocation type " << static_cast<int>(type) <<
              ", dims: " << numRows << "x" << numCols << " req.mem.: " << printBufferSize() << "Mb" << " with at least " << strBufferCapacity_ << " bytes for strings" <<  std::endl;
#endif
    if (type == ALLOCATION_TYPE::HOST_ALLOC) {
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

DenseMatrix<const char*>* DenseMatrix<const char*>::vectorTranspose() const {
    assert((this->numRows == 1 || this->numCols == 1) && "no-op transpose for vectors only");

    auto transposed = DataObjectFactory::create<DenseMatrix<const char*>>(this->getNumCols(), this->getNumRows(),
                                                                        this->getValuesSharedPtr());
    return transposed;
}

// Convert to an integer to print uint8_t values as numbers
// even if they fall into the range of special ASCII characters.
template <> void DenseMatrix<unsigned char>::printValue(std::ostream & os, unsigned char val) const
{
    os << static_cast<unsigned int>(val);
}
template <> void DenseMatrix<signed char>::printValue(std::ostream & os, signed char val) const
{
    os << static_cast<int>(val);
}

#ifdef USE_CUDA
template <typename ValueType>
void DenseMatrix<ValueType>::cuda2host() {
    if(!values)
        alloc_shared_values();
    if(cuda_ptr)
        CHECK_CUDART(cudaMemcpy(values.get(), cuda_ptr.get(), numRows*numCols*sizeof(ValueType), cudaMemcpyDeviceToHost));
    else
        throw std::runtime_error("trying to cudaMemcpy from null ptr\n");
    cuda_dirty = false;
}

template <typename ValueType>
void DenseMatrix<ValueType>::host2cuda() {
    if (!cuda_ptr) {
        alloc_shared_cuda_buffer();
    }
    CHECK_CUDART(cudaMemcpy(cuda_ptr.get(), values.get(), numRows*numCols*sizeof(ValueType), cudaMemcpyHostToDevice));
    host_dirty = false;
}


template<typename ValueType>
void DenseMatrix<ValueType>::alloc_shared_cuda_buffer(std::shared_ptr<ValueType> src, size_t offset) {
    if(src) {
//#ifndef NDEBUG
//        std::ios state(nullptr);
//        state.copyfmt(std::cout);
//        std::cout << "Increasing refcount on cuda buffer " << src.get() << " of size" << printBufferSize()
//                <<  "Mb from << " << src.use_count() << " to ";
//        std::cout.copyfmt(state);
//#endif
        this->cuda_ptr = std::shared_ptr<ValueType>(src, src.get() + offset);

//#ifndef NDEBUG
//        std::cout  << src.use_count() << "\n new cuda_ptr's use_count: " << cuda_ptr.use_count() << std::endl;
//#endif
    }
    else {
        auto* dev_ptr = new ValueType;
#ifndef NDEBUG
        if(this->rowSkip != this->numCols) {
            std::cerr << "Warning: setting rowSkip to numCols in alloc_shared_cuda_buffer" << std::endl;
            rowSkip = numCols;
        }
//        std::cout << "Allocating new cuda buffer of size " << printBufferSize() << "Mb at address ";
//        std::ios state(nullptr);
//        state.copyfmt(std::cout);
//        std::cout << "addressof dev_ptr: " << &dev_ptr << std::endl;
#endif
        CHECK_CUDART(cudaMalloc(reinterpret_cast<void **>(&dev_ptr), this->bufferSize()));
        this->cuda_ptr = std::shared_ptr<ValueType>(dev_ptr, CudaDeleter<ValueType>());
//#ifndef NDEBUG
//        std::cout << "addressof dev_ptr after cudaMalloc: " << &dev_ptr << std::endl;
//        std::cout.copyfmt(state);
//        std::cout << cuda_ptr.get() << " use count: " << cuda_ptr.use_count() << std::endl;
//#endif
    }
}
#else
    template<typename ValueType>
    void DenseMatrix<ValueType>::alloc_shared_cuda_buffer(std::shared_ptr<ValueType> src, size_t offset) { }
#endif // USE_CUDA

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
