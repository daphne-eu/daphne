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

#ifndef SRC_RUNTIME_LOCAL_DATASTRUCTURES_CSRMATRIX_H
#define SRC_RUNTIME_LOCAL_DATASTRUCTURES_CSRMATRIX_H

#include <runtime/local/datastructures/BaseMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>

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
class CSRMatrix : public BaseMatrix {
    ValueType * values;
    size_t * colIdxs;
    size_t * rowOffsets;
    
    // Grant DataObjectFactory::create access to the private constructors.
    template<class DataType, typename ... ArgTypes>
    friend DataType * DataObjectFactory::create(ArgTypes ...);
    
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
            BaseMatrix(maxNumRows, numCols),
            values(new ValueType[maxNumNonZeros]),
            colIdxs(new size_t[maxNumNonZeros]),
            rowOffsets(new size_t[numRows + 1])
    {
        if(zero) {
            memset(values, 0, maxNumNonZeros * sizeof(ValueType));
            memset(colIdxs, 0, maxNumNonZeros * sizeof(size_t));
            memset(rowOffsets, 0, (numRows + 1) * sizeof(size_t));
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
    CSRMatrix(const CSRMatrix<ValueType> * src, size_t rowLowerIncl, size_t rowUpperExcl) {
        assert(src && "src must not be null");
        assert((rowLowerIncl < numRows) && "rowLowerIncl is out of bounds");
        assert((rowUpperExcl <= numRows) && "rowUpperExcl is out of bounds");
        assert((rowLowerIncl < rowUpperExcl) && "rowLowerIncl must be lower than rowUpperExcl");
        
        numRows = rowUpperExcl - rowLowerIncl;
        numCols = src.numCols;
        
        values = src.values;
        colIdxs = src.colIdxs;
        rowOffsets = src.rowOffsets + rowLowerIncl;
    }
    
    virtual ~CSRMatrix() {
        // Enable safe sharing.
        delete[] values;
        delete[] colIdxs;
        delete[] rowOffsets;
    }
    
public:
    void shrinkNumRows(size_t numRows) {
        assert((numRows <= this->numRows) && "numRows can only the shrinked");
        // TODO Here we could reduce the allocated size of the rowOffsets array.
        this->numRows = numRows;
    }
    
    size_t getNumNonZeros() const {
        return rowOffsets[numRows] - rowOffsets[0];
    }
    
    size_t getNumNonZeros(size_t rowIdx) const {
        assert((rowIdx < numRows) && "rowIdx is out of bounds");
        return rowOffsets[rowIdx + 1] - rowOffsets[rowIdx];
    }
    
    void shrinkNumNonZeros(size_t numNonZeros) {
        assert((numNonZeros <= getNumNonZeros()) && "numNonZeros can only the shrinked");
        // TODO Here we could reduce the allocated size of the values and
        // colIdxs arrays.
    }
    
    ValueType * getValues() {
        return values;
    }
    
    const ValueType * getValues() const {
        return values;
    }
    
    ValueType * getValues(size_t rowIdx) {
        assert((rowIdx < numRows) && "rowIdx is out of bounds");
        return values + rowOffsets[rowIdx];
    }
    
    const ValueType * getValues(size_t rowIdx) const {
        return const_cast<CSRMatrix<ValueType> *>(this)->getValues(rowIdx);
    }
    
    size_t * getColIdxs() {
        return colIdxs;
    }
    
    const size_t * getColIdxs() const {
        return colIdxs;
    }
    
    size_t * getColIdxs(size_t rowIdx) {
        assert((rowIdx < numRows) && "rowIdx is out of bounds");
        return colIdxs + rowOffsets[rowIdx];
    }
    
    const size_t * getColIdxs(size_t rowIdx) const {
        return const_cast<CSRMatrix<ValueType> *>(this)->getColIdxs(rowIdx);
    }
    
    size_t * getRowPtrs() {
        return rowOffsets;
    }
    
    const size_t * getRowPtrs() const {
        return rowOffsets;
    }
    
};

#endif //SRC_RUNTIME_LOCAL_DATASTRUCTURES_DENSEMATRIX_H