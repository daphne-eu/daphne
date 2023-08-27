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

      if(zero) {
          memset(values.get(), 0, maxNumNonZeros * sizeof(ValueType));
          memset(rowIdxs.get(), 0, maxNumNonZeros * sizeof(size_t));
          memset(columnOffsets.get(), 0, (numCols + 1) * sizeof(size_t));
      }

      std::cout << "CSC Matrix" << '\n';


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


  ValueType get(size_t rowIdx, size_t colIdx) const override {
    return values.get()[0];
  }

  void set(size_t rowIdx, size_t colIdx, ValueType value) override {
      assert((rowIdx < numRows) && "rowIdx is out of bounds");
      assert((colIdx < numCols) && "colIdx is out of bounds");
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
