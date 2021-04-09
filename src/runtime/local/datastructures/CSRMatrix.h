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

#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/Matrix.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>

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
     * @brief The maximum number of non-zero values this matrix was allocated
     * to accommodate.
     */
    size_t maxNumNonZeros;
    
    std::shared_ptr<ValueType> values;
    std::shared_ptr<size_t> colIdxs;
    std::shared_ptr<size_t> rowOffsets;
    
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
            maxNumNonZeros(maxNumNonZeros),
            values(new ValueType[maxNumNonZeros]),
            colIdxs(new size_t[maxNumNonZeros]),
            rowOffsets(new size_t[numRows + 1])
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
            Matrix<ValueType>(rowUpperExcl - rowLowerIncl, src->numCols)
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
    
public:
    
    void shrinkNumRows(size_t numRows) {
        assert((numRows <= this->numRows) && "numRows can only the shrinked");
        // TODO Here we could reduce the allocated size of the rowOffsets array.
        this->numRows = numRows;
    }
    
    size_t getNumNonZeros() const {
        return rowOffsets.get()[numRows] - rowOffsets.get()[0];
    }
    
    size_t getNumNonZeros(size_t rowIdx) const {
        assert((rowIdx < numRows) && "rowIdx is out of bounds");
        return rowOffsets.get()[rowIdx + 1] - rowOffsets.get()[rowIdx];
    }
    
    void shrinkNumNonZeros(size_t numNonZeros) {
        assert((numNonZeros <= getNumNonZeros()) && "numNonZeros can only the shrinked");
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
                os << oneRow[c];
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
    
};

template <typename ValueType>
std::ostream & operator<<(std::ostream & os, const CSRMatrix<ValueType> & obj)
{
    obj.print(os);
    return os;
}

#endif //SRC_RUNTIME_LOCAL_DATASTRUCTURES_DENSEMATRIX_H