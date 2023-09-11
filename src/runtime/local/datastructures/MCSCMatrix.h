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
 #include <cmath>
 #include <cassert>
 #include <cstddef>
 #include <cstring>


 template<typename ValueType>
 class MCSCMatrix : public Matrix<ValueType>{

   using Matrix<ValueType>::numRows;
   using Matrix<ValueType>::numCols;
   size_t numColsAllocated;
   bool isColAllocatedBefore;
   size_t maxNumNonZeros;
   std::shared_ptr<std::shared_ptr<ValueType>[]> values;
   std::shared_ptr<std::shared_ptr<size_t>[]> rowIdxs;
   std::shared_ptr<size_t[]> valueSizes;
   std::shared_ptr<size_t[]> allocatedColSizes;

   // Grant DataObjectFactory access to the private constructors and
   // destructors.
   template<class DataType, typename ... ArgTypes>
   friend DataType * DataObjectFactory::create(ArgTypes ...);
   template<class DataType>
   friend void DataObjectFactory::destroy(const DataType * obj);


   MCSCMatrix(size_t numRows, size_t maxNumCols, size_t maxNumNonZeros, bool zero) :
    Matrix<ValueType>(numRows, maxNumCols),
    numColsAllocated(maxNumCols),
    isColAllocatedBefore(false),
    maxNumNonZeros(maxNumNonZeros),
    values(new std::shared_ptr<ValueType>[numCols], std::default_delete<std::shared_ptr<ValueType>[]>()),
    rowIdxs(new std::shared_ptr<size_t>[numCols], std::default_delete<std::shared_ptr<size_t>[]>()),
    valueSizes(new size_t[numCols], std::default_delete<size_t[]>()),
    allocatedColSizes(new size_t[numCols], std::default_delete<size_t[]>())

    {
        assert(numCols != 0 && "Number of columns must not be zero.");
        size_t baselineColSize = std::ceil(1.1 * maxNumNonZeros / numCols);
        for(size_t i = 0; i < numCols; i++){
            values.get()[i] = std::shared_ptr<ValueType>(new ValueType[baselineColSize], std::default_delete<ValueType[]>());
            rowIdxs.get()[i] = std::shared_ptr<size_t>(new size_t[baselineColSize], std::default_delete<size_t[]>());
            if(zero){
                memset(values.get()[i].get(), 0, baselineColSize * sizeof(ValueType));
                memset(rowIdxs.get()[i].get(), 0, baselineColSize * sizeof(size_t));
            }
            allocatedColSizes.get()[i] = baselineColSize;
        }
        if(zero){
            memset(valueSizes.get(), 0, numCols * sizeof(size_t));
        }
    }

    MCSCMatrix(const MCSCMatrix<ValueType> * src, size_t colLowerIncl, size_t colUpperExcl) :
    Matrix<ValueType>(src->numRows, colUpperExcl - colLowerIncl),
    numColsAllocated(src->numColsAllocated - colLowerIncl),
    isColAllocatedBefore(colLowerIncl > 0)
    {
        maxNumNonZeros = src->maxNumNonZeros;
        values = std::shared_ptr<std::shared_ptr<ValueType>[]>(src->values, src->values.get() + colLowerIncl);
        rowIdxs = std::shared_ptr<std::shared_ptr<size_t>[]>(src->rowIdxs, src->rowIdxs.get() + colLowerIncl);
        valueSizes = std::shared_ptr<size_t[]>(src->valueSizes, src->valueSizes.get() + colLowerIncl);
        allocatedColSizes = std::shared_ptr<size_t[]>(src->allocatedColSizes, src->allocatedColSizes.get() + colLowerIncl);
    }

    virtual ~MCSCMatrix() {
        // nothing to do
    }

    void reallocateColumn(size_t colIdx) {
        // 1. Determine the new size for the column.
        const float growthFactor = 1.5;
        size_t currentSize = allocatedColSizes.get()[colIdx];
        size_t newSize = static_cast<size_t>(currentSize * growthFactor);

        // 2. Allocate new arrays.
        std::shared_ptr<ValueType> newColumnValues(new ValueType[newSize], std::default_delete<ValueType[]>());
        std::shared_ptr<size_t> newColumnRowIdxs(new size_t[newSize], std::default_delete<size_t[]>());

        // 3. Copy old data.
        memcpy(newColumnValues.get(), values.get()[colIdx].get(), currentSize * sizeof(ValueType));
        memcpy(newColumnRowIdxs.get(), rowIdxs.get()[colIdx].get(), currentSize * sizeof(size_t));

        // Initialize the rest of the new space with zeros.
        memset(newColumnValues.get() + currentSize, 0, (newSize - currentSize) * sizeof(ValueType));
        memset(newColumnRowIdxs.get() + currentSize, 0, (newSize - currentSize) * sizeof(size_t));

        // 4. Update the pointers.
        values.get()[colIdx] = newColumnValues;
        rowIdxs.get()[colIdx] = newColumnRowIdxs;

        // 5. Update the allocatedColSizes.
        allocatedColSizes.get()[colIdx] = newSize;
    }



public:

  size_t getNumRows() const{
    return numRows;
  }

  size_t getNumCols() const{
      return numCols;
  }

  size_t * getAllocatedColSizes() const {
      return allocatedColSizes.get();
  }

  size_t getMaxNumNonZeros() const {
      return maxNumNonZeros;
  }

  size_t getNumNonZeros() const {
      size_t total = 0;
      for(size_t i = 0; i < numCols; i++){
          total += valueSizes.get()[i];
      }
      return total;
  }

  size_t getNumNonZeros(size_t colIdx) const {
      assert((colIdx < numCols) && "colIdx is out of bounds");
      return valueSizes.get()[colIdx];
  }

  size_t * getAllNumNonZeros() const {
      return valueSizes.get();
  }

  ValueType * getValues(size_t colIdx) {
    return values.get()[colIdx].get();
  }

  const ValueType * getValues(size_t colIdx) const {
      return values.get()[colIdx].get();
  }

  size_t * getRowIdxs(size_t colIdx) {
      assert((colIdx < numCols) && "colIdx is out of bounds");
      return rowIdxs.get()[colIdx].get();
  }

  const size_t* getRowIdxs(size_t colIdx) const {
    assert((colIdx < numCols) && "colIdx is out of bounds");
    return rowIdxs.get()[colIdx].get();
  }


  ValueType get(size_t rowIdx, size_t colIdx) const override {
    // Check if column index is valid
    assert(colIdx < numCols && "colIdx is out of bounds");

    // Get the rows and values for the specified column
    size_t* currentRowIdxs = rowIdxs.get()[colIdx].get();
    ValueType* currentValues = values.get()[colIdx].get();

    // The length of the current column's arrays (i.e., number of non-zero elements in this column)
    size_t length = valueSizes.get()[colIdx];

    // Search for the row index in the current column's rowIdxs
    for(size_t i = 0; i < length; i++) {
        if (currentRowIdxs[i] == rowIdx) {
            return currentValues[i];  // Return the value if found
        }
    }

    // If row index wasn't found, the value is zero
    return static_cast<ValueType>(0);
  }

  void set(size_t rowIdx, size_t colIdx, ValueType value) override {
    assert(rowIdx < numRows && colIdx < numCols && "Indices out of bounds");

    // Retrieve the column's arrays and its current size.
    ValueType* columnValues = values.get()[colIdx].get();
    size_t* columnRowIdxs = rowIdxs.get()[colIdx].get();
    size_t columnSize = valueSizes.get()[colIdx];

    // If the value is zero and previously was non-zero, remove it.
    if (value == 0) {
        bool found = false;
        for (size_t i = 0; i < columnSize; i++) {
            if (columnRowIdxs[i] == rowIdx) {
                found = true;
                // Shift the remaining values and row indices.
                for (size_t j = i; j < columnSize - 1; j++) {
                    columnValues[j] = columnValues[j + 1];
                    columnRowIdxs[j] = columnRowIdxs[j + 1];
                }
                // Reset the last (non-active) value and row index.
                columnValues[columnSize - 1] = 0;
                columnRowIdxs[columnSize - 1] = 0;

                valueSizes.get()[colIdx]--;
                break;
            }
        }
        if (!found) {
            // The value was already zero, nothing to do.
            return;
        }
    } else { // The value is non-zero
        bool updated = false;
        for (size_t i = 0; i < columnSize; i++) {
            if (columnRowIdxs[i] == rowIdx) {
                // Update the existing value.
                columnValues[i] = value;
                updated = true;
                break;
            }
        }

        if (!updated) {
            // The value is new, ensure we have space to add it.
            if (columnSize >= allocatedColSizes.get()[colIdx]) {
                // Reallocation logic...
                // (Increase the allocated space and copy the old values)
                reallocateColumn(colIdx);
                // Also adjust the column pointers after reallocation.
                columnValues = values.get()[colIdx].get();
                columnRowIdxs = rowIdxs.get()[colIdx].get();
            }
            // Append the new value and row index at the end.
            columnValues[columnSize] = value;
            columnRowIdxs[columnSize] = rowIdx;

            // Increment the size for this column.
            valueSizes.get()[colIdx]++;
        }
    }
  }

  void prepareAppend() override {
      // Not needed for MCSC
  }

  void append(size_t rowIdx, size_t colIdx, ValueType value) override {
    assert(rowIdx < numRows && colIdx < numCols && "Indices out of bounds");

    // If the value is zero, just return.
    if (value == 0) return;

    ValueType* columnValues = values.get()[colIdx].get();
    size_t* columnRowIdxs = rowIdxs.get()[colIdx].get();
    size_t columnSize = valueSizes.get()[colIdx];

    // If we've used up all the allocated space, reallocate more memory.
    if (columnSize >= allocatedColSizes.get()[colIdx]) {
        // Reallocation logic...
        reallocateColumn(colIdx);

        // Also adjust the column pointers after reallocation.
        columnValues = values.get()[colIdx].get();
        columnRowIdxs = rowIdxs.get()[colIdx].get();
    }

    // Find the right position to insert the new value.
    size_t position = 0;
    while (position < columnSize && columnRowIdxs[position] < rowIdx) {
        position++;
    }

    // Shift the values and row indices to the right from the found position.
    for (size_t i = columnSize; i > position; i--) {
        columnValues[i] = columnValues[i - 1];
        columnRowIdxs[i] = columnRowIdxs[i - 1];
    }

    // Insert the new value and its row index.
    columnValues[position] = value;
    columnRowIdxs[position] = rowIdx;

    // Increase the size for this column.
    valueSizes.get()[colIdx]++;
  }

  void finishAppend() override {
    // Not needed for MCSC
  }

  bool isView() const {
    return (numColsAllocated > numCols || isColAllocatedBefore);
  }

  void printValue(std::ostream & os, ValueType val) const {
    switch (ValueTypeUtils::codeFor<ValueType>) {
        case ValueTypeCode::SI8 : os << static_cast<int32_t>(val); break;
        case ValueTypeCode::UI8 : os << static_cast<uint32_t>(val); break;
        default : os << val; break;
    }
  }

  void print(std::ostream & os) const override {
      os << "MCSCMatrix(" << numRows << 'x' << numCols << ", "
         << ValueTypeUtils::cppNameFor<ValueType> << ')' << std::endl;

      ValueType * oneRow = new ValueType[numCols];

      for (size_t r = 0; r < numRows; r++) {
          // Clear the row array to start with all zeros
          memset(oneRow, 0, numCols * sizeof(ValueType));
          // Iterate over each column to populate the values for the current row
          for (size_t c = 0; c < numCols; c++) {
              const size_t colNumNonZeros = valueSizes.get()[c];
              const size_t * currentRowIdxs = rowIdxs.get()[c].get();
              const ValueType * colValues = values.get()[c].get();

              for(size_t i = 0; i < colNumNonZeros; i++) {
                  if (currentRowIdxs[i] == r) {
                      oneRow[c] = colValues[i];
                      break;
                  }
              }
          }
          for(size_t c = 0; c < numCols; c++) {
              printValue(os, oneRow[c]);
              if (c < numCols - 1)
                  os << ' ';
          }
          os << std::endl;
      }
      delete[] oneRow;
  }



  MCSCMatrix* sliceRow(size_t colLowerIncl, size_t colUpperExcl) const override {
    throw std::runtime_error("MCSCMatrix does not support sliceRow yet");
  }

  MCSCMatrix* sliceCol(size_t colLowerIncl, size_t colUpperExcl) const override {
    assert(colUpperExcl<numCols && "Indices out of bounds");
    return DataObjectFactory::create<MCSCMatrix>(this, colLowerIncl, colUpperExcl);

  }

  MCSCMatrix* slice(size_t rl, size_t ru, size_t cl, size_t cu) const override {
      throw std::runtime_error("MCSCMatrix does not support slice yet");
  }

  size_t serialize(std::vector<char> &buf) const override{
    // MCSRMatrix is not yet integrated into DaphneSerializer
    return 0;
  }

 };

 template <typename ValueType>
 std::ostream & operator<<(std::ostream & os, const MCSCMatrix<ValueType> & obj)
 {
     obj.print(os);
     return os;
 }
