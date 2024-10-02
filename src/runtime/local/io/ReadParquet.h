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

#include <type_traits>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <queue>
#include <sstream>

#include <arrow/api.h>
#include <arrow/csv/api.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/api.h>
#include <parquet/arrow/reader.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes> struct ReadParquet {
    static void apply(DTRes *&res, const char *filename, size_t numRows, size_t numCols) = delete;
    static void apply(DTRes *&res, const char *filename, size_t numRows, size_t numCols,
                      ValueTypeCode *schema) = delete;
    static void apply(DTRes *&res, const char *filename, size_t numRows, size_t numCols, ssize_t numNonZeros,
                      bool sorted = true) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes> void readParquet(DTRes *&res, const char *filename, size_t numRows, size_t numCols) {
    ReadParquet<DTRes>::apply(res, filename, numRows, numCols);
}

template <class DTRes>
void readParquet(DTRes *&res, const char *filename, size_t numRows, size_t numCols, ValueTypeCode *schema) {
    ReadParquet<DTRes>::apply(res, filename, numRows, numCols, schema);
}

template <class DTRes>
void readParquet(DTRes *&res, const char *filename, size_t numRows, size_t numCols, ssize_t numNonZeros,
                 bool sorted = true) {
    ReadParquet<DTRes>::apply(res, filename, numRows, numCols, numNonZeros, sorted);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

inline struct File *arrowToCsv(const char *filename) {
    arrow::MemoryPool *pool = arrow::default_memory_pool();
    arrow::fs::LocalFileSystem file_system;
    std::shared_ptr<arrow::io::RandomAccessFile> input = file_system.OpenInputFile(filename).ValueOrDie();

    std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
    if (!(parquet::arrow::OpenFile(input, pool, &arrow_reader).ok()))
        throw std::runtime_error("Could not open Parquet file");

    std::shared_ptr<arrow::Table> table;
    if (!(arrow_reader->ReadTable(&table)).ok())
        throw std::runtime_error("Could not read Parquet table");

    auto output = arrow::io::BufferOutputStream::Create().ValueOrDie();
    if (!(arrow::csv::WriteCSV(*table, arrow::csv::WriteOptions::Defaults(), output.get())).ok())
        throw std::runtime_error("Could not write from Parquet to CSV format");

    auto finishResult = output->Finish();

    auto csv = finishResult.ValueOrDie()->ToString();
    void *ccsv = csv.data();

    FILE *buf = fmemopen(ccsv, csv.size(), "r");
    struct File *file = openMemFile(buf);
    if (getFileLine(file) == -1) // Parquet has headers, readCsv does not expect that.
        throw std::runtime_error("arrowToCsv: getFileLine failed");

    return file;
}

// ----------------------------------------------------------------------------
// Frame
// ----------------------------------------------------------------------------

template <> struct ReadParquet<Frame> {
    static void apply(Frame *&res, const char *filename, size_t numRows, size_t numCols, ValueTypeCode *schema) {
        struct File *file = arrowToCsv(filename);
        readCsvFile<Frame>(res, file, numRows, numCols, ',', schema);
        closeFile(file);
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct ReadParquet<CSRMatrix<VT>> {
    static void apply(CSRMatrix<VT> *&res, const char *filename, size_t numRows, size_t numCols, ssize_t numNonZeros,
                      bool sorted = true) {
        struct File *file = arrowToCsv(filename);
        readCsvFile<CSRMatrix<VT>>(res, file, numRows, numCols, ',', numNonZeros, sorted);
        closeFile(file);
    }
};

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct ReadParquet<DenseMatrix<VT>> {
    static void apply(DenseMatrix<VT> *&res, const char *filename, size_t numRows, size_t numCols) {
        struct File *file = arrowToCsv(filename);
        readCsvFile<DenseMatrix<VT>>(res, file, numRows, numCols, ',');
        closeFile(file);
    }
};
