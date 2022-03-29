/*
 * Copyright 2022 The DAPHNE Consortium
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

#ifndef MM_IO_H
#define MM_IO_H

#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/Handle.h>
#include <runtime/local/datastructures/ValueTypeCode.h>
#include <runtime/local/io/MMFile.h>
#include <vector>
#include <algorithm>
#include <queue>

typedef char MM_typecode[4];

char *mm_typecode_to_str(MM_typecode matcode);

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes> struct ReadMM {
  static void apply(DTRes *&res, const char *filename) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes>
void readMM(DTRes *&res, const char *filename) {
  ReadMM<DTRes>::apply(res, filename);
}

template <typename VT> struct ReadMM<DenseMatrix<VT>> {
  static void apply(DenseMatrix<VT> *&res, const char *filename){
    MMFile<VT> mmfile(filename);
    if(res == nullptr)
      res = DataObjectFactory::create<DenseMatrix<VT>>(
        mmfile.numberRows(),
        mmfile.numberCols(),
        mmfile.entryCount() != mmfile.numberCols() * mmfile.numberRows()
      );
    VT *valuesRes = res->getValues();
    for (auto &entry : mmfile)
      valuesRes[entry.row * mmfile.numberCols() + entry.col] = entry.val;
  }
};

template <typename VT> struct ReadMM<CSRMatrix<VT>> {
  static void apply(CSRMatrix<VT> *&res, const char *filename){
    MMFile<VT> mmfile(filename);

    using entry_t = typename MMFile<VT>::Entry;
    std::priority_queue<entry_t, std::vector<entry_t>, std::greater<>>
      entry_queue;

    for(auto &entry : mmfile) entry_queue.emplace(entry);

    if(res == nullptr)
      res = DataObjectFactory::create<CSRMatrix<VT>>(
        mmfile.numberRows(),
        mmfile.numberCols(),
        entry_queue.size(),
        false
      );

    auto *rowOffsets = res->getRowOffsets();
    rowOffsets[0] = 0;
    auto *colIdxs = res->getColIdxs();
    auto *values = res->getValues();
    size_t currValIdx = 0;
    size_t rowIdx = 0;
    while(!entry_queue.empty()){
      auto& entry = entry_queue.top();
      while(rowIdx < entry.row) {
          rowOffsets[rowIdx + 1] = currValIdx;
          rowIdx++;
      }
      values[currValIdx] = entry.val;
      colIdxs[currValIdx] = entry.col;
      currValIdx++;
      entry_queue.pop();
    }
    while(rowIdx < mmfile.numberRows()) {
        rowOffsets[rowIdx + 1] = currValIdx;
        rowIdx++;
    }
  }
};

template <> struct ReadMM<Frame> {
  static void apply(Frame *&res, const char *filename){
    MMFile<double> mmfile(filename);

    if(res == nullptr){
      ValueTypeCode *types = new ValueTypeCode[mmfile.numberCols()];
      for(size_t i = 0; i<mmfile.numberCols(); i++)
        types[i] = mmfile.elementType();

      res = DataObjectFactory::create<Frame>(
        mmfile.numberRows(),
        mmfile.numberCols(),
        types, nullptr,
        mmfile.entryCount() != mmfile.numberCols() * mmfile.numberRows()
      );
    }
    uint8_t ** rawFrame = new uint8_t*[mmfile.numberCols()];
    for(size_t i = 0; i<mmfile.numberCols(); i++)
      rawFrame[i] = reinterpret_cast<uint8_t *>(res->getColumnRaw(i));
    for(auto& entry : mmfile)
      if (mmfile.elementType() == ValueTypeCode::SI64)
        reinterpret_cast<int64_t *>(rawFrame[entry.col])[entry.row] = (int64_t)entry.val;
      else
        reinterpret_cast<double *>(rawFrame[entry.col])[entry.row] = entry.val;
  }
};

#endif // MM_IO_H