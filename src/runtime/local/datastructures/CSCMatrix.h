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
//#include <stdio.h>
#include <cassert>
#include <cstddef>
#include <cstring>

/**
 * @brief A sparse matrix in Compressed Sparse Column (CSC) format.
 *
 * This matrix implementation is backed by three contiguous arrays. The
 * `values` array contains all non-zero values in the matrix. For each of these
 * non-zero values, the `rowIdxs` array contains the number of the row.
 * Finally, the `colOffsets` array contains for each column in the matrix the
 * offset at which the corresponding entries can be found in the `values` and
 * `rowIdxs` arrays. Additionally, the `colOffsets` array ends with the offset
 * to the first element after the valid elements in the `values` and `rowIdxs`
 * arrays.
 *
 * Each instance of this class might represent a sub-matrix of another
 * `CSCMatrix`. Thus, to traverse the matrix by column, you can safely go via the
 * `colOffsets`, but for traversing the matrix by non-zero value, you must
 * start at `values[colOffsets[0]`.
 */


template<typename ValueType>
class CSCMatrix : public Matrix<ValueType>{

  /**
 * @brief Detailed description of the class variables in the CSCMatrix class.
 *
 * @param numRows Inherited from the Matrix class, representing the number of rows in the matrix.
 * @param numCols Inherited from the Matrix class, representing the number of columns in the matrix.
 * @param maxNumNonZeros The maximum number of non-zero values that the matrix can hold.
 * @param numColumnsAllocated The number of columns for which memory has been allocated.
 * @param isColumnAllocatedBefore A flag indicating whether a column has been allocated before.
 * @param values A shared pointer to the array holding the non-zero values of the matrix.
 * @param rowIdxs A shared pointer to the array holding the row indices corresponding to the non-zero values.
 * @param columnOffsets A shared pointer to the array holding the offsets for each column in the values and rowIdxs arrays.
 * @param lastAppendedColumnIdx The index of the last column to which a value was appended.
 */

  using Matrix<ValueType>::numRows;
  using Matrix<ValueType>::numCols;
  size_t maxNumNonZeros;
  size_t numColumnsAllocated;
  bool isColumnAllocatedBefore;
  std::shared_ptr<ValueType[]> values;
  std::shared_ptr<size_t[]> rowIdxs;
  std::shared_ptr<size_t[]> columnOffsets;
  size_t lastAppendedColumnIdx;

  // Grant DataObjectFactory access to the private constructors and
  // destructors.
  template<class DataType, typename ... ArgTypes>
  friend DataType * DataObjectFactory::create(ArgTypes ...);
  template<class DataType>
  friend void DataObjectFactory::destroy(const DataType * obj);

  /**
 * @brief Constructs a CSCMatrix with specified dimensions and initializes memory.
 *
 * The constructor allocates memory for storing non-zero values, row indices, and column offsets.
 * If the `zero` flag is set, it initializes all allocated memory to zero.
 *
 * @param numRows Number of rows in the matrix.
 * @param numCols Number of columns in the matrix.
 * @param maxNumNonZeros Maximum number of non-zero values the matrix can store.
 * @param zero Flag indicating whether to initialize the memory to zero.
 */

  CSCMatrix(size_t numRows, size_t numCols, size_t maxNumNonZeros, bool zero):
    Matrix<ValueType>(numRows, numCols),
    maxNumNonZeros(maxNumNonZeros),
    numColumnsAllocated(numCols),
    isColumnAllocatedBefore(false),
    values(new ValueType[maxNumNonZeros], std::default_delete<ValueType[]>()),
    rowIdxs(new size_t[maxNumNonZeros], std::default_delete<size_t[]>()),
    columnOffsets(new size_t[numCols + 1], std::default_delete<size_t[]>()),
    lastAppendedColumnIdx(0)
  {

    if(zero) {
        memset(values.get(), 0, maxNumNonZeros * sizeof(ValueType));
        memset(rowIdxs.get(), 0, maxNumNonZeros * sizeof(size_t));
        memset(columnOffsets.get(), 0, (numCols + 1) * sizeof(size_t));
    }

  }

  /**
 * @brief Constructs a view (sub-matrix) of the original CSCMatrix based on column boundaries.
 *
 * This constructor creates a view of the original matrix, using shared pointers to reference the
 * original data without copying. It defines a sub-matrix spanning from `colLowerIncl` to `colUpperExcl`
 * columns, inclusive of the lower boundary and exclusive of the upper boundary.
 *
 * @param src Pointer to the original CSCMatrix.
 * @param colLowerIncl Lower column boundary (inclusive).
 * @param colUpperExcl Upper column boundary (exclusive).
 */

    CSCMatrix(const CSCMatrix<ValueType>* src, size_t colLowerIncl, size_t colUpperExcl):
      Matrix<ValueType>(src -> numRows, colUpperExcl - colLowerIncl),
      numColumnsAllocated(src->numColumnsAllocated - colLowerIncl),
      isColumnAllocatedBefore(colLowerIncl>0),
      lastAppendedColumnIdx(0)
    {
        assert(src && "src must not be null");
        assert((colLowerIncl < src->numCols) && "colLowerIncl is out of bounds");
        assert((colUpperExcl <= src->numCols) && "colUpperExcl is out of bounds");
        assert((colLowerIncl < colUpperExcl) && "colLowerIncl must be lower than colUpperExcl");

        maxNumNonZeros = src->maxNumNonZeros;
        values = src->values;
        rowIdxs = src->rowIdxs;  // In CSC, we store row indices
        columnOffsets = std::shared_ptr<size_t[]>(src->columnOffsets, src->columnOffsets.get() + colLowerIncl);
    }

    virtual ~CSCMatrix() {
        // nothing to do
    }

    void fillNextPosUntil(size_t nextPos, size_t colIdx) {
      if(colIdx>lastAppendedColumnIdx){
        for(size_t i = lastAppendedColumnIdx+2; i<=colIdx+1; i++) {
          columnOffsets.get()[i] = nextPos;
        }
        lastAppendedColumnIdx = colIdx;
      }
    }

public:

  size_t getNumRows() const{
    return numRows;
  }

  size_t getNumCols() const{
    return numCols;
  }

  size_t getMaxNumNonZeros() const{
    return maxNumNonZeros;
  }

  size_t getNumNonZeros() const {
      return columnOffsets.get()[numCols] - columnOffsets.get()[0];
  }

  size_t getNumNonZeros(size_t columnIdx) const {
      //assert((rowIdx < numRows) && "rowIdx is out of bounds");
      return columnOffsets.get()[columnIdx + 1] - columnOffsets.get()[columnIdx];
  }

  // return values array
  //**************************************************
  ValueType * getValues() {
      return values.get();
  }

  const ValueType * getValues() const {
      return values.get();
  }

  //  return pointer to column
  ValueType * getValues(size_t columnIdx) {
      return values.get() + columnOffsets.get()[columnIdx];
  }

  const ValueType * getValues(size_t columnIdx) const {
      return const_cast<CSCMatrix<ValueType> *>(this)->getValues(columnIdx);
  }
  //**************************************************


  // return row indexes
  //**************************************************
  size_t * getRowIdxs() {
      return rowIdxs.get();
  }

  const size_t * getRowIdxs() const {
      return rowIdxs.get();
  }

  size_t * getRowIdxs(size_t columnIdx) {
      // We allow equality here to enable retrieving a pointer to the end.
      return rowIdxs.get() + columnOffsets.get()[columnIdx];
  }

  const size_t * getRowIdxs(size_t columnIdx) const {
      return const_cast<CSCMatrix<ValueType> *>(this)->getRowIdxs(columnIdx);
  }
  //**************************************************


  // return columnOffsets
  //**************************************************
  size_t * getColumnOffsets() {
      return columnOffsets.get();
  }

  const size_t * getColumnOffsets() const {
      return columnOffsets.get();
  }
  //**************************************************



  // get matrix cell by coordinates
  ValueType get(size_t rowIdx, size_t colIdx) const override {
    // Get the starting and ending pointers for the specified column
    size_t start = columnOffsets.get()[colIdx];
    size_t end = columnOffsets.get()[colIdx+1];

    // Search for the row index within the column's non-zero entries
    for(size_t i = start; i<end; i++){
      if(rowIdxs.get()[i] == rowIdx){
        return values.get()[i]; //Return the value if found
      }
    }

    return 0;
  }


  // set value in existing matrix
  void set(size_t rowIdx, size_t colIdx, ValueType value) override {
    assert(rowIdx < numRows && "rowIdx is out of bounds");
    assert(colIdx < numCols && "colIdx is out of bounds");

    // Find the column pointer range
    size_t colStart = columnOffsets.get()[colIdx];
    size_t colEnd = columnOffsets.get()[colIdx + 1];

    // Find the position where this rowIdx would be inserted
    auto it = std::lower_bound(rowIdxs.get() + colStart, rowIdxs.get() + colEnd, rowIdx);

    if (it != rowIdxs.get() + colEnd && *it == rowIdx) {
        // Element found, update value
        size_t pos = it - rowIdxs.get();
        if (value == 0) {
            // Remove the element if the value is zero
            for (size_t i = pos; i < colEnd - 1; ++i) {
                values.get()[i] = values.get()[i + 1];
                rowIdxs.get()[i] = rowIdxs.get()[i + 1];
            }
            // Update column pointers
            for (size_t i = colIdx + 1; i <= numCols; ++i) {
                columnOffsets.get()[i]--;
            }
        } else {
            // Update the value
            values.get()[pos] = value;
        }
    } else if (value != 0.0) {
        // No existing value, insert new element if the value is non-zero
        size_t insertPos = it - rowIdxs.get();

        // Shift the elements in the values and rowIdxs arrays
        for (size_t i = maxNumNonZeros; i > insertPos; --i) {
            values.get()[i] = values.get()[i - 1];
            rowIdxs.get()[i] = rowIdxs.get()[i - 1];
        }

        // Insert the new value and row index
        values.get()[insertPos] = value;
        rowIdxs.get()[insertPos] = rowIdx;

        // Update column pointers
        for (size_t i = colIdx + 1; i <= numCols; ++i) {
            columnOffsets.get()[i]++;
        }

    }
  }

  // Process of appending new values in matrix:
  // prepareAppend() -> append()...append() -> finishAppend()
  //**************************************************
  void prepareAppend() override {
    if(isColumnAllocatedBefore)
        // In this case, we assume that the matrix has been populated up to
        // just before this view.
        columnOffsets.get()[1] = columnOffsets.get()[0];
    else
        columnOffsets.get()[1] = columnOffsets.get()[0] = 0;
    lastAppendedColumnIdx = 0;
  }

  // Note that if this matrix is a view on a larger `CSCMatrix`, then
  // `prepareAppend`/`append`/`finishAppend` assume that the larger matrix
  // has been populated up to just before the column range of this view.
  void append(size_t rowIdx, size_t colIdx, ValueType value) override {
    assert(colIdx>=0 && "column index is out of bounds.");
    assert(colIdx<numCols && "column index is out of bounds.");
    assert(rowIdx<numRows && "row index is out of bounds.");

    //Skip zero values
    if(value==0){
      return;
    }

    const size_t nextPos = columnOffsets.get()[lastAppendedColumnIdx+1];
    if(colIdx != lastAppendedColumnIdx){
      fillNextPosUntil(nextPos, colIdx);
    }

    //Append the value and row index
    values.get()[nextPos] = value;
    rowIdxs.get()[nextPos] = rowIdx;

    //Increment the next column pointer
    columnOffsets.get()[colIdx+1]++;
  }

  void finishAppend() override {
    fillNextPosUntil(columnOffsets.get()[lastAppendedColumnIdx + 1], numCols - 1);
  }
  //**************************************************

  bool isView() const {
      return (numColumnsAllocated > numCols || isColumnAllocatedBefore);
  }

  void printValue(std::ostream & os, ValueType val) const {
    switch (ValueTypeUtils::codeFor<ValueType>) {
      case ValueTypeCode::SI8 : os << static_cast<int32_t>(val); break;
      case ValueTypeCode::UI8 : os << static_cast<uint32_t>(val); break;
      default : os << val; break;
    }
  }

  void print(std::ostream & os) const override {
    os << "CSCMatrix(" << numRows << 'x' << numCols << ", "
       << ValueTypeUtils::cppNameFor<ValueType> << ')' << std::endl;

    // First, let's cache our column data, so we don't need to fetch it repeatedly
    std::vector<ValueType *> allColumns(numCols);
    for (size_t c = 0; c < numCols; c++) {
        allColumns[c] = new ValueType[numRows];
        memset(allColumns[c], 0, numRows * sizeof(ValueType));
        const size_t colNumNonZeros = getNumNonZeros(c);
        const size_t * colRowIdxs = getRowIdxs(c);
        const ValueType * colValues = getValues(c);
        for (size_t i = 0; i < colNumNonZeros; i++) {
            allColumns[c][colRowIdxs[i]] = colValues[i];
        }
    }

    // Now we print the matrix row by row
    for (size_t r = 0; r < numRows; r++) {
        for (size_t c = 0; c < numCols; c++) {
            printValue(os, allColumns[c][r]);
            if (c < numCols - 1)
                os << ' ';
        }
        os << std::endl;
    }

    // Cleanup memory
    for (size_t c = 0; c < numCols; c++) {
        delete[] allColumns[c];
    }
  }




  CSCMatrix* sliceRow(size_t cl, size_t cu) const override {
      //return DataObjectFactory::create<CSCMatrix>(this, rl, ru);
      throw std::runtime_error("CSCMatrix does not support sliceRow yet");
  }

  CSCMatrix* sliceCol(size_t cl, size_t cu) const override {
      //throw std::runtime_error("CSCMatrix does not support sliceCol yet");
      return DataObjectFactory::create<CSCMatrix>(this, cl, cu);
  }

  CSCMatrix* slice(size_t rl, size_t ru, size_t cl, size_t cu) const override {
      throw std::runtime_error("CSCMatrix does not support slice yet");
  }



  size_t serialize(std::vector<char> &buf) const override{
    // CSCMatrix is not yet integrated into DaphneSerializer
    return 0;
  }


};

template <typename ValueType>
std::ostream & operator<<(std::ostream & os, const CSCMatrix<ValueType> & obj)
{
    obj.print(os);
    return os;
}
