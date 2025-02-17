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

#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>

#include <runtime/local/io/File.h>
#include <runtime/local/io/ReadCsvFile.h>
#include <runtime/local/io/utils.h>

#include <util/preprocessor_defs.h>

#include <type_traits>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <queue>
#include <sstream>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes> struct ReadCsv {
    static void apply(DTRes *&res, const char *filename, size_t numRows, size_t numCols, char delim, ReadOpts opt = ReadOpts()) = delete;

    static void apply(DTRes *&res, const char *filename, size_t numRows, size_t numCols, ssize_t numNonZeros,
                      bool sorted = true, ReadOpts opt = ReadOpts()) = delete;

    static void apply(DTRes *&res, const char *filename, size_t numRows, size_t numCols, char delim,
                      ValueTypeCode *schema, ReadOpts opt = ReadOpts()) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes> void readCsv(DTRes *&res, const char *filename, size_t numRows, size_t numCols, char delim, ReadOpts opt = ReadOpts()) {
    ReadCsv<DTRes>::apply(res, filename, numRows, numCols, delim, opt);
}

template <class DTRes>
void readCsv(DTRes *&res, const char *filename, size_t numRows, size_t numCols, char delim, ValueTypeCode *schema, ReadOpts opt = ReadOpts()) {
    ReadCsv<DTRes>::apply(res, filename, numRows, numCols, delim, schema, opt);
}

template <class DTRes>
void readCsv(DTRes *&res, const char *filename, size_t numRows, size_t numCols, char delim, ssize_t numNonZeros, bool sorted = true,
             ReadOpts opt = ReadOpts()) {
    ReadCsv<DTRes>::apply(res, filename, numRows, numCols, delim, numNonZeros, sorted, opt);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct ReadCsv<DenseMatrix<VT>> {
    static void apply(DenseMatrix<VT> *&res, const char *filename, size_t numRows, size_t numCols, char delim, ReadOpts opt = ReadOpts()) {
        struct File *file = openFile(filename);
        readCsvFile(res, file, numRows, numCols, delim, filename, opt);
        closeFile(file);
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct ReadCsv<CSRMatrix<VT>> {
    static void apply(CSRMatrix<VT> *&res, const char *filename, size_t numRows, size_t numCols, char delim,
                      ssize_t numNonZeros, bool sorted = true, ReadOpts opt = ReadOpts()) {
        struct File *file = openFile(filename);
        readCsvFile(res, file, numRows, numCols, delim, numNonZeros, sorted, filename, opt);
        closeFile(file);
    }
};

// ----------------------------------------------------------------------------
// Frame
// ----------------------------------------------------------------------------

template <> struct ReadCsv<Frame> {
    static void apply(Frame *&res, const char *filename, size_t numRows, size_t numCols, char delim,
                      ValueTypeCode *schema, ReadOpts opt = ReadOpts()) {
        struct File *file = openFile(filename);
        readCsvFile(res, file, numRows, numCols, delim, schema, filename, opt);
        closeFile(file);
    }
};
