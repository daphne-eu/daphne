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
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/Handle.h>
#include <runtime/local/kernels/DistributedCaller.h>

#include <runtime/local/io/File.h>
#include <runtime/local/io/utils.h>

#include <util/preprocessor_defs.h>

#include <type_traits>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <queue>
#include <fstream>
#include <limits>
#include <sstream>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes> struct ReadDaphne {
  static void apply(DTRes *&res, const char *filename, size_t numRows, size_t numCols) = delete;

  static void apply(DTRes *&res, const char *filename, size_t numRows, size_t numCols,
                    ssize_t numNonZeros, bool sorted = true) = delete;

  static void apply(DTRes *&res, const char *filename, size_t numRows, size_t numCols,
                    ValueTypeCode *schema) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes>
void readDaphne(DTRes *&res, const char *filename, size_t numRows, size_t numCols,
             ) {
  ReadDaphne<DTRes>::apply(res, filename, numRows, numCols);
}

template <class DTRes>
void readCsv(DTRes *&res, const char *filename, size_t numRows, size_t numCols,
             ValueTypeCode *schema) {
  ReadCsv<DTRes>::apply(res, filename, numRows, numCols, schema);
}

template <class DTRes>
void readCsv(DTRes *&res, const char *filename, size_t numRows, size_t numCols,
             ssize_t numNonZeros, bool sorted = true) {
    ReadCsv<DTRes>::apply(res, filename, numRows, numCols, numNonZeros, sorted);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct ReadDpahne<DenseMatrix<VT>> {
  static void apply(DenseMatrix<VT> *&res, const char *filename, size_t numRows,
                    size_t numCols) {
    struct File *file = openFile(filename);
    readDaphneFile(res, file, numRows, numCols);
    closeFile(file);
  }
};

// ----------------------------------------------------------------------------
// CSRMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct ReadDaphne<CSRMatrix<VT>> {
    static void apply(CSRMatrix<VT> *&res, const char *filename, size_t numRows,
                      size_t numCols, ssize_t numNonZeros, bool sorted = true) {
        struct File *file = openFile(filename);
        readDaphneFile(res, file, numRows, numCols, numNonZeros, sorted);
        closeFile(file);
    }
};


// ----------------------------------------------------------------------------
// Frame
// ----------------------------------------------------------------------------

template <> struct ReadDpahne<Frame> {
  static void apply(Frame *&res, const char *filename, size_t numRows,
                    size_t numCols, ValueTypeCode *schema) {
    struct File *file = openFile(filename);
    readDaphneFile(res, file, numRows, numCols, schema);
    closeFile(file);
  }
};


// ----------------------------------------------------------------------------
// Handle
// ----------------------------------------------------------------------------

template <class DT> struct ReadDaphne<Handle<DT>> {
  static void apply(Handle<DT> *&res, const char *filename, 
                      size_t numRows, size_t numCols) {
    struct File *file = openFile(filename);
    readDaphneFile(res, file, numRows, numCols, delim);
    closeFile(file);
  }
};
