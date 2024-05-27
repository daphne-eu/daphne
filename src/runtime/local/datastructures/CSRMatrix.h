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

#pragma once

#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/Matrix.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>

#include <algorithm>
#include <iostream>
#include <memory>

#include <cassert>
#include <cstddef>
#include <cstring>

/**
 * @brief A sparse matrix in Compressed Sparse Row (CSR) format.
 * 
 * This matrix implementation is backed by three contiguous arrays. The
 * `values` array contains all non-zero values in the matrix. For each of these
 * non-zero values, the `colIdxs` array contains the number of the column.
 * Finally, the `rowOffsets` array contains for each row in the matrix the
 * offset at which the corresponding entries can be found in the `values` and
 * `colIdxs` arrays. Additionally, the `rowOffsets` array ends with the offset
 * to the first element after the valid elements in the `values` and `colIdxs`
 * arrays.
 * 
 * Each instance of this class might represent a sub-matrix of another
 * `CSRMatrix`. Thus, to traverse the matrix by row, you can safely go via the
 * `rowOffsets`, but for traversing the matrix by non-zero value, you must
 * start at `values[rowOffsets[0]`.
 */
template<typename ValueType>
class CSRMatrix : public Matrix<ValueType> {
    // `using`, so that we do not need to prefix each occurrence of these
    // fields from the super-classes.
    using Matrix<ValueType>::numRows;
    using Matrix<ValueType>::numCols;
    
    /**
     * @brief The number of rows allocated starting from `rowOffsets`. This can
     * differ from `numRows` if this `CSRMatrix` is a view on a larger
     * `CSRMatrix`.
     */
    size_t numRowsAllocated;
    
    bool isRowAllocatedBefore;
    
    /**
     * @brief The maximum number of non-zero values this matrix was allocated
     * to accommodate.
     */
    size_t maxNumNonZeros;
    
    std::shared_ptr<ValueType> values;
    std::shared_ptr<size_t> colIdxs;
    std::shared_ptr<size_t> rowOffsets;
    
    size_t lastAppendedRowIdx;

    // Grant DataObjectFactory access to the private constructors and
    // destructors.
    template<class DataType, typename ... ArgTypes>
    friend DataType * DataObjectFactory::create(ArgTypes ...);
    template<class DataType>
    friend void DataObjectFactory::destroy(const DataType * obj);
    
    /**
     * @brief Creates a `CSRMatrix` and allocates enough memory for the
     * specified size in the internal `values`, `colIdxs`, and `rowOffsets`
     * arrays.
     * 
     * @param maxNumRows The maximum number of rows.
     * @param numCols The exact number of columns.
     * @param maxNumNonZeros The maximum number of non-zeros in the matrix.
     * @param zero Whether the allocated memory of the internal arrays shall be
     * initialized to zeros (`true`), or be left uninitialized (`false`).
     */
    CSRMatrix(size_t maxNumRows, size_t numCols, size_t maxNumNonZeros, bool zero) : 
            Matrix<ValueType>(maxNumRows, numCols),
            numRowsAllocated(maxNumRows),
            isRowAllocatedBefore(false),
            maxNumNonZeros(maxNumNonZeros),
            values(new ValueType[maxNumNonZeros], std::default_delete<ValueType[]>()),
            colIdxs(new size_t[maxNumNonZeros], std::default_delete<size_t[]>()),
            rowOffsets(new size_t[numRows + 1], std::default_delete<size_t[]>()),
            lastAppendedRowIdx(0)
    {
        if(zero) {
            memset(values.get(), 0, maxNumNonZeros * sizeof(ValueType));
            memset(colIdxs.get(), 0, maxNumNonZeros * sizeof(size_t));
            memset(rowOffsets.get(), 0, (numRows + 1) * sizeof(size_t));
        }
    }
    
    /**
     * @brief Creates a `CSRMatrix` around a sub-matrix of another `CSRMatrix`
     * without copying the data.
     * 
     * @param src The other `CSRMatrix`.
     * @param rowLowerIncl Inclusive lower bound for the range of rows to extract.
     * @param rowUpperExcl Exclusive upper bound for the range of rows to extract.
     */
    CSRMatrix(const CSRMatrix<ValueType> * src, size_t rowLowerIncl, size_t rowUpperExcl) :
            Matrix<ValueType>(rowUpperExcl - rowLowerIncl, src->numCols),
            numRowsAllocated(src->numRowsAllocated - rowLowerIncl),
            isRowAllocatedBefore(rowLowerIncl > 0),
            lastAppendedRowIdx(0)
    {
        assert(src && "src must not be null");
        assert((rowLowerIncl < src->numRows) && "rowLowerIncl is out of bounds");
        assert((rowUpperExcl <= src->numRows) && "rowUpperExcl is out of bounds");
        assert((rowLowerIncl < rowUpperExcl) && "rowLowerIncl must be lower than rowUpperExcl");
        
        maxNumNonZeros = src->maxNumNonZeros;
        values = src->values;
        colIdxs = src->colIdxs;
        rowOffsets = std::shared_ptr<size_t>(src->rowOffsets, src->rowOffsets.get() + rowLowerIncl);
    }
    
    virtual ~CSRMatrix() {
        // nothing to do
    }
    
    void fillNextPosUntil(size_t nextPos, size_t rowIdx) {
        if(rowIdx > lastAppendedRowIdx) {
            for(size_t r = lastAppendedRowIdx + 2; r <= rowIdx + 1; r++)
                rowOffsets.get()[r] = nextPos;
            lastAppendedRowIdx = rowIdx;
        }
    }
    
public:

    template<typename NewValueType>
    using WithValueType = CSRMatrix<NewValueType>;
    
    void shrinkNumRows(size_t numRows) {
        assert((numRows <= this->numRows) && "numRows can only the shrinked");
        // TODO Here we could reduce the allocated size of the rowOffsets array.
        this->numRows = numRows;
    }
    
    size_t getMaxNumNonZeros() const {
        return maxNumNonZeros;
    }
    size_t getNumNonZeros() const {
        return rowOffsets.get()[numRows] - rowOffsets.get()[0];
    }
    
    size_t getNumNonZeros(size_t rowIdx) const {
        assert((rowIdx < numRows) && "rowIdx is out of bounds");
        return rowOffsets.get()[rowIdx + 1] - rowOffsets.get()[rowIdx];
    }
    
    void shrinkNumNonZeros(size_t numNonZeros) {
        assert((numNonZeros <= getNumNonZeros()) && "numNonZeros can only be shrinked");
        // TODO Here we could reduce the allocated size of the values and
        // colIdxs arrays.
    }

    ValueType * getValues() {
        return values.get();
    }
    
    const ValueType * getValues() const {
        return values.get();
    }
    
    ValueType * getValues(size_t rowIdx) {
        // We allow equality here to enable retrieving a pointer to the end.
        assert((rowIdx <= numRows) && "rowIdx is out of bounds");
        return values.get() + rowOffsets.get()[rowIdx];
    }
    
    const ValueType * getValues(size_t rowIdx) const {
        return const_cast<CSRMatrix<ValueType> *>(this)->getValues(rowIdx);
    }
    
    size_t * getColIdxs() {
        return colIdxs.get();
    }
    
    const size_t * getColIdxs() const {
        return colIdxs.get();
    }
    
    size_t * getColIdxs(size_t rowIdx) {
        // We allow equality here to enable retrieving a pointer to the end.
        assert((rowIdx <= numRows) && "rowIdx is out of bounds");
        return colIdxs.get() + rowOffsets.get()[rowIdx];
    }

    const size_t * getColIdxs(size_t rowIdx) const {
        return const_cast<CSRMatrix<ValueType> *>(this)->getColIdxs(rowIdx);
    }

    size_t * getRowOffsets() {
        return rowOffsets.get();
    }

    const size_t * getRowOffsets() const {
        return rowOffsets.get();
    }

    ValueType get(size_t rowIdx, size_t colIdx) const override {
        assert((rowIdx < numRows) && "rowIdx is out of bounds");
        assert((colIdx < numCols) && "colIdx is out of bounds");
        
        const size_t * rowColIdxsBeg = getColIdxs(rowIdx);
        const size_t * rowColIdxsEnd = getColIdxs(rowIdx + 1);
        const size_t * ptrExpected = std::lower_bound(rowColIdxsBeg, rowColIdxsEnd, colIdx);

        if(ptrExpected == rowColIdxsEnd || *ptrExpected != colIdx)
            // No entry for the given coordinates present.
            return ValueType(0);
        else
            // Entry for the given coordinates present.
            return getValues(rowIdx)[ptrExpected - rowColIdxsBeg];
    }
    
    void set(size_t rowIdx, size_t colIdx, ValueType value) override {
        assert((rowIdx < numRows) && "rowIdx is out of bounds");
        assert((colIdx < numCols) && "colIdx is out of bounds");
        
        size_t * rowColIdxsBeg = getColIdxs(rowIdx);
        size_t * rowColIdxsEnd = getColIdxs(rowIdx + 1);
        const size_t * ptrExpected = std::lower_bound(rowColIdxsBeg, rowColIdxsEnd, colIdx);
        const size_t posExpected = ptrExpected - rowColIdxsBeg;
        
        const size_t posEnd = colIdxs.get() + rowOffsets.get()[numRowsAllocated] - rowColIdxsBeg;
        ValueType * rowValuesBeg = getValues(rowIdx);
        
        if(ptrExpected == rowColIdxsEnd || *ptrExpected != colIdx) {
            // No entry for the given coordinates present.
            if(value == ValueType(0))
                return; // do nothing
            else {
                // Create gap.
                // TODO We might want to reallocate here to ensure that enough
                // space is allocated.
                if(posEnd)
                    for(size_t pos = posEnd; pos > posExpected; pos--) {
                        rowValuesBeg[pos] = rowValuesBeg[pos - 1];
                        rowColIdxsBeg[pos] = rowColIdxsBeg[pos - 1];
                    }
                // Insert given value and column index into the gap.
                rowValuesBeg[posExpected] = value;
                rowColIdxsBeg[posExpected] = colIdx;
                // Update rowOffsets.
                for(size_t r = rowIdx + 1; r <= numRowsAllocated; r++)
                    rowOffsets.get()[r]++;
            }
        }
        else {
            // Entry for the given coordinates present.
            if(value == ValueType(0)) {
                // Close gap.
                for(size_t pos = posExpected; pos < posEnd; pos++) {
                    rowValuesBeg[pos] = rowValuesBeg[pos + 1];
                    rowColIdxsBeg[pos] = rowColIdxsBeg[pos + 1];
                }
                // Update rowOffsets.
                // TODO We might want to shrink the arrays here.
                for(size_t r = rowIdx + 1; r <= numRowsAllocated; r++)
                    rowOffsets.get()[r]--;
            }
            else
                // Simply overwrite the existing value.
                rowValuesBeg[posExpected] = value;
        }
    }
    
    void prepareAppend() override {
        if(isRowAllocatedBefore)
            // In this case, we assume that the matrix has been populated up to
            // just before this view.
            rowOffsets.get()[1] = rowOffsets.get()[0];
        else
            rowOffsets.get()[1] = rowOffsets.get()[0] = 0;
        lastAppendedRowIdx = 0;
    }
    
    // Note that if this matrix is a view on a larger `CSRMatrix`, then
    // `prepareAppend`/`append`/`finishAppend` assume that the larger matrix
    // has been populated up to just before the row range of this view.
    void append(size_t rowIdx, size_t colIdx, ValueType value) override {
        assert((rowIdx < numRows) && "rowIdx is out of bounds");
        assert((colIdx < numCols) && "colIdx is out of bounds");
        
        if(value == ValueType(0))
            return;
        
        const size_t nextPos = rowOffsets.get()[lastAppendedRowIdx + 1];
        fillNextPosUntil(nextPos, rowIdx);
        
        values.get()[nextPos] = value;
        colIdxs.get()[nextPos] = colIdx;
        rowOffsets.get()[rowIdx + 1]++;
    }
    
    void finishAppend() override {
        fillNextPosUntil(rowOffsets.get()[lastAppendedRowIdx + 1], numRows - 1);
    }

    bool isView() const {
        return (numRowsAllocated > numRows || isRowAllocatedBefore);
    }
    
    void printValue(std::ostream & os, ValueType val) const {
      switch (ValueTypeUtils::codeFor<ValueType>) {
        case ValueTypeCode::SI8 : os << static_cast<int32_t>(val); break;
        case ValueTypeCode::UI8 : os << static_cast<uint32_t>(val); break;
        default : os << val; break;
      }
    }

    void print(std::ostream & os) const override {
        os << "CSRMatrix(" << numRows << 'x' << numCols << ", "
                << ValueTypeUtils::cppNameFor<ValueType> << ')' << std::endl;
        // Note that, in general, the values within one row might not be sorted
        // by column index. Thus, the following is a little complicated.
        ValueType * oneRow = new ValueType[numCols];
        for (size_t r = 0; r < numRows; r++) {
            memset(oneRow, 0, numCols * sizeof(ValueType));
            const size_t rowNumNonZeros = getNumNonZeros(r);
            const size_t * rowColIdxs = getColIdxs(r);
            const ValueType * rowValues = getValues(r);
            for(size_t i = 0; i < rowNumNonZeros; i++)
                oneRow[rowColIdxs[i]] = rowValues[i];
            for(size_t c = 0; c < numCols; c++) {
                printValue(os, oneRow[c]);
                if (c < numCols - 1)
                    os << ' ';
            }
            os << std::endl;
        }
        delete[] oneRow;
    }
    
    /**
     * @brief Prints the internal arrays of this matrix.
     * 
     * Meant to be used for testing and debugging only. Note that this method
     * works even if the internal state of the matrix is invalid, e.g., due to
     * uninitialized row offsets or column indexes.
     * 
     * @param os The stream to print to.
     */
    void printRaw(std::ostream & os) const {
        os << "CSRMatrix(" << numRows << 'x' << numCols << ", "
                << ValueTypeUtils::cppNameFor<ValueType> << ')' << std::endl;
        os << "maxNumNonZeros: \t" << maxNumNonZeros << std::endl;
        os << "values: \t";
        for(size_t i = 0; i < maxNumNonZeros; i++)
            os << values.get()[i] << ", ";
        os << std::endl;
        os << "colIdxs: \t";
        for(size_t i = 0; i < maxNumNonZeros; i++)
            os << colIdxs.get()[i] << ", ";
        os << std::endl;
        os << "rowOffsets: \t";
        for(size_t i = 0; i <= numRows; i++)
            os << rowOffsets.get()[i] << ", ";
        os << std::endl;
    }

    CSRMatrix* sliceRow(size_t rl, size_t ru) const override {
        return DataObjectFactory::create<CSRMatrix>(this, rl, ru);
    }

    CSRMatrix* sliceCol(size_t cl, size_t cu) const override {
        // TODO add boundary validation when implementing
        throw std::runtime_error("CSRMatrix does not support sliceCol yet");
    }

    CSRMatrix* slice(size_t rl, size_t ru, size_t cl, size_t cu) const override {
        // TODO add boundary validation when implementing
        throw std::runtime_error("CSRMatrix does not support slice yet");
    }

    size_t bufferSize() {
        return this->getNumItems() * sizeof(ValueType);
    }

    bool operator==(const CSRMatrix<ValueType> & rhs) const {
        // Note that we do not use the generic `get` interface to matrices here since
        // this operator is meant to be used for writing tests for, besides others,
        // those generic interfaces.

        if(this == &rhs)
            return true;
        
        const size_t numRows = this->getNumRows();
        const size_t numCols = this->getNumCols();
        
        if(numRows != rhs.getNumRows() || numCols != rhs.getNumCols())
            return false;
        
        const ValueType * valuesBegLhs = this->getValues(0);
        const ValueType * valuesEndLhs = this->getValues(numRows);
        const ValueType * valuesBegRhs = rhs.getValues(0);
        const ValueType * valuesEndRhs = rhs.getValues(numRows);
        
        const size_t nnzLhs = valuesEndLhs - valuesBegLhs;
        const size_t nnzRhs = valuesEndRhs - valuesBegRhs;
        
        if(nnzLhs != nnzRhs)
            return false;
        
        if(valuesBegLhs != valuesBegRhs)
            if(memcmp(valuesBegLhs, valuesBegRhs, nnzLhs * sizeof(ValueType)))
                return false;
        
        const size_t * colIdxsBegLhs = this->getColIdxs(0);
        const size_t * colIdxsBegRhs = rhs.getColIdxs(0);
        
        if(colIdxsBegLhs != colIdxsBegRhs)
            if(memcmp(colIdxsBegLhs, colIdxsBegRhs, nnzLhs * sizeof(size_t)))
                return false;
        
        return true;
    }

    size_t serialize(std::vector<char> &buf) const override ;
};

template <typename ValueType>
std::ostream & operator<<(std::ostream & os, const CSRMatrix<ValueType> & obj)
{
    obj.print(os);
    return os;
}
