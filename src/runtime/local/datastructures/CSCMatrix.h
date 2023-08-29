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
#include <stdio.h>
#include <cassert>
#include <cstddef>
#include <cstring>

template<typename ValueType>
class CSCMatrix : public Matrix<ValueType>{

  using Matrix<ValueType>::numRows;
  using Matrix<ValueType>::numCols;
  size_t maxNumNonZeros;
  size_t numColumnsAllocated;
  bool isColumnAllocatedBefore;
  std::shared_ptr<ValueType> values;
  std::shared_ptr<size_t> rowIdxs;
  std::shared_ptr<size_t> columnOffsets;
  size_t lastAppendedColumnIdx;

  // Grant DataObjectFactory access to the private constructors and
  // destructors.
  template<class DataType, typename ... ArgTypes>
  friend DataType * DataObjectFactory::create(ArgTypes ...);
  template<class DataType>
  friend void DataObjectFactory::destroy(const DataType * obj);

public:
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

      values.get()[0] = 10; values.get()[1] = 20; values.get()[2] = 30;
      values.get()[3] = 50; values.get()[4] = 40; values.get()[5] = 60;
      values.get()[6] = 70; values.get()[7] = 80;

      rowIdxs.get()[0] = 0; rowIdxs.get()[1] = 0; rowIdxs.get()[2] = 1;
      rowIdxs.get()[3] = 2; rowIdxs.get()[4] = 1; rowIdxs.get()[5] = 2;
      rowIdxs.get()[6] = 2; rowIdxs.get()[7] = 3;

      columnOffsets.get()[0] = 0; columnOffsets.get()[1] = 1; columnOffsets.get()[2] = 3;
      columnOffsets.get()[3] = 4; columnOffsets.get()[4] = 6; columnOffsets.get()[5] = 7;
      columnOffsets.get()[6] = 8;

      if(zero) {
          memset(values.get(), 0, maxNumNonZeros * sizeof(ValueType));
          memset(rowIdxs.get(), 0, maxNumNonZeros * sizeof(size_t));
          memset(columnOffsets.get(), 0, (numCols + 1) * sizeof(size_t));
      }

      //std::cout << "CSC Matrix" << '\n';
    }

    CSCMatrix(const CSCMatrix<ValueType>* orig, size_t cl, size_t cu):
      Matrix<ValueType>(cu - cl, orig->numCols)
    {

    }

    virtual ~CSCMatrix() {
        // nothing to do
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
        //maxNumNonZeros++;
    }
  }

  void prepareAppend() override {

  }

  // Note that if this matrix is a view on a larger `CSCMatrix`, then
  // `prepareAppend`/`append`/`finishAppend` assume that the larger matrix
  // has been populated up to just before the row range of this view.
  void append(size_t rowIdx, size_t colIdx, ValueType value) override {

  }

  void finishAppend() override {
  }




  void print(std::ostream & os) const override {

  }



  CSCMatrix* sliceRow(size_t rl, size_t ru) const override {
      return DataObjectFactory::create<CSCMatrix>(this, rl, ru);
      //throw std::runtime_error("CSCMatrix does not support sliceCol yet");
  }

  CSCMatrix* sliceCol(size_t cl, size_t cu) const override {
      throw std::runtime_error("CSCMatrix does not support sliceCol yet");
  }

  CSCMatrix* slice(size_t rl, size_t ru, size_t cl, size_t cu) const override {
      throw std::runtime_error("CSCMatrix does not support slice yet");
  }



  size_t serialize(std::vector<char> &buf) const override{
    return 0;
  }


};
