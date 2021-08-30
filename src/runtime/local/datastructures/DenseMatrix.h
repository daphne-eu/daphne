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
 * obtain a pointer to the corresponsing cell in the next row.
 */
template <typename ValueType>
class DenseMatrix : public Matrix<ValueType>
{
    // `using`, so that we do not need to prefix each occurrence of these
    // fields from the super-classes.
    using Matrix<ValueType>::numRows;
    using Matrix<ValueType>::numCols;
    
    size_t rowSkip;
    std::shared_ptr<ValueType> values;
    ValueType* cuda_ptr = nullptr;
	mutable bool cuda_dirty = false;
	mutable bool host_dirty = false;

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
    DenseMatrix(size_t maxNumRows, size_t numCols, bool zero, ALLOCATION_TYPE type = ALLOCATION_TYPE::HOST_ALLOC) :
            Matrix<ValueType>(maxNumRows, numCols),
            rowSkip(numCols),
            values(new ValueType[maxNumRows * numCols]),
            lastAppendedRowIdx(0),
            lastAppendedColIdx(0)
    {
//#ifndef NDEBUG
//    	std::cerr << "creating dense matrix of allocation type " << static_cast<int>(type) <<
//    	" and dims: " << numRows << "x" << numCols << std::endl;
//#endif
    	if (type == ALLOCATION_TYPE::HOST_ALLOC) {
//    		values = std::make_shared<ValueType>(maxNumRows*numCols);
//            values(new ValueType[maxNumRows * numCols]);
    		host_dirty = true;
    		if(zero)
    			memset(values.get(), 0, maxNumRows * numCols * sizeof(ValueType));
    	}
#ifdef USE_CUDA
    	else if (type == ALLOCATION_TYPE::CUDA_ALLOC) {
    		cudaAlloc();
    		cuda_dirty = true;
    	}
#endif
    	else {
    		throw std::runtime_error("Unknown allocation type: " + std::to_string(static_cast<int>(type)));
    	}
    }
    
    /**
     * @brief Creates a `DenseMatrix` around an existing array of values
     * without copying the data.
     * 
     * @param numRows The exact number of rows.
     * @param numCols The exact number of columns.
     * @param values The existing array of values.
     */
    DenseMatrix(size_t numRows, size_t numCols, ValueType * values) :
            Matrix<ValueType>(numRows, numCols),
            rowSkip(numCols),
            values(values),
            lastAppendedRowIdx(0),
            lastAppendedColIdx(0)
    {
        assert(values && "values must not be null");
        host_dirty = true;
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
    DenseMatrix(const DenseMatrix * src, size_t rowLowerIncl, size_t rowUpperExcl, size_t colLowerIncl, size_t colUpperExcl) :
            Matrix<ValueType>(rowUpperExcl - rowLowerIncl, colUpperExcl - colLowerIncl),
            lastAppendedRowIdx(0),
            lastAppendedColIdx(0)
    {
        assert(src && "src must not be null");
        assert((rowLowerIncl < src->numRows) && "rowLowerIncl is out of bounds");
        assert((rowUpperExcl <= src->numRows) && "rowUpperExcl is out of bounds");
        assert((rowLowerIncl < rowUpperExcl) && "rowLowerIncl must be lower than rowUpperExcl");
        assert((colLowerIncl < src->numCols) && "colLowerIncl is out of bounds");
        assert((colUpperExcl <= src->numCols) && "colUpperExcl is out of bounds");
        assert((colLowerIncl < colUpperExcl) && "colLowerIncl must be lower than colUpperExcl");
        
        rowSkip = src->rowSkip;
        values = std::shared_ptr<ValueType>(src->values, src->values.get() + rowLowerIncl * src->rowSkip + colLowerIncl);
        host_dirty = src->host_dirty;
        cuda_dirty = src->cuda_dirty;
        cuda_ptr = src->cuda_ptr;
    }
    
    virtual ~DenseMatrix();

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
    
public:
    
    void shrinkNumRows(size_t numRows) {
        assert((numRows <= this->numRows) && "number of rows can only the shrinked");
        // TODO Here we could reduce the allocated size of the values array.
        this->numRows = numRows;
    }
    
    [[nodiscard]] size_t getRowSkip() const {
        return rowSkip;
    }

    const ValueType * getValues() const
    {
#ifdef USE_CUDA
    	if (cuda_dirty) {
    		cuda2host();
    	}
    	host_dirty = true;
#endif
    	return values.get();
    };

    ValueType * getValues()
    {
		return const_cast<ValueType*>(values.get());
    }

#ifdef USE_CUDA
    const ValueType* getValuesCUDA() const {
		return const_cast<DenseMatrix*>(this)->getValuesCUDA();
    }

    ValueType* getValuesCUDA() {
//#ifndef NDEBUG
//    	std::cerr << "getValuesCUDA " << cuda_ptr << " cuda_dirty=" << cuda_dirty << " host_dirty=" << host_dirty << std::endl;
//#endif
    	if(host_dirty) {
    		host2cuda();
    	}
//#ifndef NDEBUG
//    	else
//    		std::cerr << "skipping allocation =)" << std::endl;
//#endif
    	cuda_dirty = true;
    	return cuda_ptr;
    }
#endif

    std::shared_ptr<ValueType> getValuesSharedPtr() {
        return values;
    }
    
    ValueType get(size_t rowIdx, size_t colIdx) const override {
        return values.get()[pos(rowIdx, colIdx)];
    }
    
    void set(size_t rowIdx, size_t colIdx, ValueType value) override {
        values.get()[pos(rowIdx, colIdx)] = value;
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
        size_t i = 0;
        if ((cuda_ptr && cuda_dirty) || !values) {
//#ifndef NDEBUG
//        	std::cerr << "cuda_ptr=" << cuda_ptr<<std::endl;
//        	std::cerr << "cuda_dirty=" << cuda_dirty << std::endl;
//#endif
		  	cuda2host();
        }
        for (size_t r = 0; r < numRows; r++) {
            for (size_t c = 0; c < numCols; c++) {
                os << values.get()[i + c];
                if (c < numCols - 1)
                    os << ' ';
            }
            os << std::endl;
            i += rowSkip;
        }
    }

    DenseMatrix<ValueType>* slice(size_t rl, size_t ru) {
        return slice(rl, ru, 0, numCols);
    }

    DenseMatrix<ValueType>* slice(size_t rl, size_t ru, size_t cl, size_t cu) {
        return new DenseMatrix<ValueType>(this, rl, ru, cl, cu);
    }

#ifdef USE_CUDA
    void cudaAlloc();
    void host2cuda();
    void cuda2host();

    void cudaAlloc() const {
    	const_cast<DenseMatrix*>(this)->cudaAlloc();
    }

    void host2cuda() const {
    	const_cast<DenseMatrix*>(this)->host2cuda();
    }
    void cuda2host() const {
    	const_cast<DenseMatrix*>(this)->cuda2host();
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
