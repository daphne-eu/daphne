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

#ifndef SRC_RUNTIME_LOCAL_DATASTRUCTURES_DENSEMATRIX_H
#define SRC_RUNTIME_LOCAL_DATASTRUCTURES_DENSEMATRIX_H

#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/Matrix.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>

#include <iostream>
#include <memory>

#include <cassert>
#include <cstddef>
#include <cstring>

// TODO DenseMatrix should not be concerned about CUDA.

/**
 * @brief A dense matrix implementation.
 *
 * This matrix implementation is backed by a single dense array of values. The
 * values are arranged in row-major fashion. That is, the array contains all
 * values in the first row, followed by all values in the second row, etc.
 *
 * Each instance of this class might represent a sub-matrix of another
 * `DenseMatrix`. Thus, in general, the row skip (see `getRowSkip()`) needs to
 * be added to a pointer to a particular cell in the `values` array in order to
 * obtain a pointer to the corresponding cell in the next row.
 */
template <typename ValueType>
class DenseMatrix : public Matrix<ValueType>
{
    // `using`, so that we do not need to prefix each occurrence of these
    // fields from the super-classes.
    using Matrix<ValueType>::numRows;
    using Matrix<ValueType>::numCols;
    
    size_t rowSkip;
    std::shared_ptr<ValueType[]> values{};
    std::shared_ptr<ValueType> cuda_ptr{};
    uint32_t deleted = 0;

    mutable bool cuda_dirty = false;
    mutable bool host_dirty = false;
    mutable bool cuda_buffer_current = false;
    mutable bool host_buffer_current = false;

    size_t lastAppendedRowIdx;
    size_t lastAppendedColIdx;
    
    // Grant DataObjectFactory access to the private constructors and
    // destructors.
    template<class DataType, typename ... ArgTypes>
    friend DataType * DataObjectFactory::create(ArgTypes ...);
    template<class DataType>
    friend void DataObjectFactory::destroy(const DataType * obj);

    /**
     * @brief Creates a `DenseMatrix` and allocates enough memory for the
     * specified maximum size in the `values` array.
     *
     * @param maxNumRows The maximum number of rows.
     * @param numCols The exact number of columns.
     * @param zero Whether the allocated memory of the `values` array shall be
     * initialized to zeros (`true`), or be left uninitialized (`false`).
     */
    DenseMatrix(size_t maxNumRows, size_t numCols, bool zero, ALLOCATION_TYPE type = ALLOCATION_TYPE::HOST_ALLOC);
    
    /**
     * @brief Creates a `DenseMatrix` around an existing array of values
     * without copying the data.
     *
     * @param numRows The exact number of rows.
     * @param numCols The exact number of columns.
     * @param values A `std::shared_ptr` to an existing array of values.
     */
    DenseMatrix(size_t numRows, size_t numCols, std::shared_ptr<ValueType[]>& values
                , std::shared_ptr<ValueType> cuda_ptr_ = nullptr) :
            Matrix<ValueType>(numRows, numCols),
            rowSkip(numCols),
            values(values),
            cuda_ptr(cuda_ptr_),
            lastAppendedRowIdx(0),
            lastAppendedColIdx(0)
    {
#ifdef USE_CUDA
#ifndef NDEBUG
            std::cout << "Increasing refcount on cuda buffer " << cuda_ptr.get() << " of size" << printBufferSize()
                    <<  "Mb to " << cuda_ptr.use_count() << std::endl;
#endif
#endif
    }

    /**
     * @brief Creates a `DenseMatrix` around a sub-matrix of another
     * `DenseMatrix` without copying the data.
     *
     * @param src The other dense matrix.
     * @param rowLowerIncl Inclusive lower bound for the range of rows to extract.
     * @param rowUpperExcl Exclusive upper bound for the range of rows to extract.
     * @param colLowerIncl Inclusive lower bound for the range of columns to extract.
     * @param colUpperExcl Exclusive upper bound for the range of columns to extract.
     */
    DenseMatrix(const DenseMatrix * src, size_t rowLowerIncl, size_t rowUpperExcl, size_t colLowerIncl, size_t colUpperExcl);

#if defined USE_CUDA && !defined NDEBUG
    ~DenseMatrix()
    {
        if(cuda_ptr.use_count() > 1)
            std::cout << "decreasing use_count of DenseMatrix (cuda_ptr=" << cuda_ptr.get() << " of size " <<
                   printBufferSize() << "Mb to " << cuda_ptr.use_count() << std::endl;
        else if (cuda_ptr.use_count() > 0)
            std::cout << "removing DenseMatrix (cuda_ptr=" << cuda_ptr.get() << " of size " <<
                    printBufferSize() << "Mb" << std::endl;
    }
#else
    ~DenseMatrix() override = default;
#endif

    [[nodiscard]] size_t pos(size_t rowIdx, size_t colIdx) const {
        assert((rowIdx < numRows) && "rowIdx is out of bounds");
        assert((colIdx < numCols) && "colIdx is out of bounds");
        return rowIdx * rowSkip + colIdx;
    }
    
    void fillZeroUntil(size_t rowIdx, size_t colIdx) {
        if(rowSkip == numCols || lastAppendedRowIdx == rowIdx) {
            const size_t startPosIncl = pos(lastAppendedRowIdx, lastAppendedColIdx) + 1;
            const size_t endPosExcl = pos(rowIdx, colIdx);
            if(startPosIncl < endPosExcl)
                memset(values.get() + startPosIncl, 0, (endPosExcl - startPosIncl) * sizeof(ValueType));
        }
        else {
            ValueType * v = values.get() + lastAppendedRowIdx * rowSkip;
            memset(v + lastAppendedColIdx + 1, 0, (numCols - lastAppendedColIdx - 1) * sizeof(ValueType));
            v += rowSkip;
            for(size_t r = lastAppendedRowIdx + 1; r < rowIdx; r++) {
                memset(v, 0, numCols * sizeof(ValueType));
                v += rowSkip;
            }
            if(colIdx)
                memset(v, 0, (colIdx - 1) * sizeof(ValueType));
        }
    }

    void printValue(std::ostream & os, ValueType val) const;

    void alloc_shared_values(std::shared_ptr<ValueType[]> src = nullptr, size_t offset = 0);

    void alloc_shared_cuda_buffer(std::shared_ptr<ValueType> src = nullptr, size_t offset = 0);

public:

    void shrinkNumRows(size_t numRows) {
        assert((numRows <= this->numRows) && "number of rows can only the shrunk");
        // TODO Here we could reduce the allocated size of the values array.
        this->numRows = numRows;
    }
    
    [[nodiscard]] size_t getRowSkip() const {
        return rowSkip;
    }

    const ValueType * getValues() const
    {
        if(!values)
            const_cast<DenseMatrix*>(this)->alloc_shared_values();
#ifdef USE_CUDA
        if (cuda_dirty || (!host_buffer_current && cuda_buffer_current)) {
            cuda2host();
        }
        host_buffer_current = true;
#endif
        return values.get();
    }

    ValueType * getValues()
    {
        if(!values)
            alloc_shared_values();

#ifdef USE_CUDA
        if (cuda_dirty || (!host_buffer_current && cuda_buffer_current)) {
            cuda2host();
        }
        cuda_buffer_current = false;
        host_buffer_current = true;
        host_dirty = true;
#endif
        return values.get();
    }

#ifdef USE_CUDA
    const ValueType* getValuesCUDA() const {
        if(!cuda_ptr)
            const_cast<DenseMatrix*>(this)->alloc_shared_cuda_buffer();

        if(host_dirty || (!cuda_buffer_current && host_buffer_current)) {
            host2cuda();
        }
        cuda_buffer_current = true;
        return cuda_ptr.get();
    }

    ValueType* getValuesCUDA() {
        if(!cuda_ptr)
            alloc_shared_cuda_buffer();

        if(host_dirty || (!cuda_buffer_current && host_buffer_current)) {
            host2cuda();
        }
        cuda_dirty = true;
        cuda_buffer_current = true;
        host_buffer_current = false;
        return cuda_ptr.get();
    }

    [[maybe_unused]] bool isBufferDirty(ALLOCATION_TYPE type) const {
        switch(type) {
            case ALLOCATION_TYPE::CUDA_ALLOC:
                return cuda_dirty;
            case ALLOCATION_TYPE::HOST_ALLOC:
            default:
                return host_dirty;
        }
    }

    [[maybe_unused]] bool isBufferCurrent(ALLOCATION_TYPE type) const {
        switch(type) {
            case ALLOCATION_TYPE::CUDA_ALLOC:
                return cuda_buffer_current;
            case ALLOCATION_TYPE::HOST_ALLOC:
            default:
                return host_buffer_current;
        }
    }

    std::shared_ptr<ValueType> getCUDAValuesSharedPtr() const {
        return cuda_ptr;
    }
    
    [[maybe_unused]] void printCUDAValuesSharedPtrUseCount() const {
        std::ios state(nullptr);
        state.copyfmt(std::cout);
        std::cout << "CudaPtr " << cuda_ptr.get() << " use_count: " << cuda_ptr.use_count() << std::endl;
        std::cout.copyfmt(state);
    }
#endif

    std::shared_ptr<ValueType[]> getValuesSharedPtr() const {
        return values;
    }
    
    ValueType get(size_t rowIdx, size_t colIdx) const override {
        return getValues()[pos(rowIdx, colIdx)];
    }
    
    void set(size_t rowIdx, size_t colIdx, ValueType value) override {
        auto vals = getValues();
        vals[pos(rowIdx, colIdx)] = value;
    }
    
    void prepareAppend() override {
        values.get()[0] = ValueType(0);
        lastAppendedRowIdx = 0;
        lastAppendedColIdx = 0;
    }
    
    void append(size_t rowIdx, size_t colIdx, ValueType value) override {
        // Set all cells since the last one that was appended to zero.
        fillZeroUntil(rowIdx, colIdx);
        // Set the specified cell.
        values.get()[pos(rowIdx, colIdx)] = value;
        // Update append state.
        lastAppendedRowIdx = rowIdx;
        lastAppendedColIdx = colIdx;
    }
    
    void finishAppend() override {
        if((lastAppendedRowIdx < numRows - 1) || (lastAppendedColIdx < numCols - 1))
            append(numRows - 1, numCols - 1, ValueType(0));
    }

    void print(std::ostream & os) const override {
        os << "DenseMatrix(" << numRows << 'x' << numCols << ", "
                << ValueTypeUtils::cppNameFor<ValueType> << ')' << std::endl;
#ifdef USE_CUDA
        if ((cuda_ptr && cuda_dirty) || !values) {
              cuda2host();
        }
#endif
        for (size_t r = 0; r < numRows; r++) {
            for (size_t c = 0; c < numCols; c++) {
                printValue(os, get(r, c));
                if (c < numCols - 1)
                    os << ' ';
            }
            os << std::endl;
        }
    }

    DenseMatrix<ValueType>* slice(size_t rl, size_t ru) override {
        return slice(rl, ru, 0, numCols);
    }

    DenseMatrix<ValueType>* slice(size_t rl, size_t ru, size_t cl, size_t cu) const {
        // TODO Use DataObjFactory.
        return new DenseMatrix<ValueType>(this, rl, ru, cl, cu);
    }

    size_t bufferSize();

    size_t bufferSize() const { return const_cast<DenseMatrix*>(this)->bufferSize(); }

    DenseMatrix<ValueType>* vectorTranspose() const;

#ifdef USE_CUDA
    void cudaAlloc();
    void host2cuda();
    void cuda2host();

    void host2cuda() const {
        const_cast<DenseMatrix*>(this)->host2cuda();
    }
    void cuda2host() const {
        const_cast<DenseMatrix*>(this)->cuda2host();
    }

    float printBufferSize() {
      return static_cast<float>(bufferSize()) / (1048576);
//        return bufferSize() / (1024*1024);
    }
#endif
};

template <typename ValueType>
std::ostream & operator<<(std::ostream & os, const DenseMatrix<ValueType> & obj)
{
    obj.print(os);
    return os;
}

#endif //SRC_RUNTIME_LOCAL_DATASTRUCTURES_DENSEMATRIX_H
