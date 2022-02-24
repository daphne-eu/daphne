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
#include <runtime/local/io/MMIterator.h>
#include <vector>

typedef char MM_typecode[4];

char *mm_typecode_to_str(MM_typecode matcode);

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes> struct ReadMM {
  static void apply(DTRes *&res, const char *filename) = delete;

//   static void apply(DTRes *&res, File *file, size_t numRows, size_t numCols,
//                     ssize_t numNonZeros, bool sorted = true) = delete;

//   static void apply(DTRes *&res, File *file, size_t numRows, size_t numCols,
//                     char delim, ValueTypeCode *schema) = delete;
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
        false
      );
    VT *valuesRes = res->getValues();
    for (auto &entry : mmfile)
      valuesRes[entry.row * mmfile.numberRows() + entry.col] = entry.val;
    return;
  }
};
#endif // MM_IO_H