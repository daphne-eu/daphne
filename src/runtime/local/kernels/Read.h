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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_READ_H
#define SRC_RUNTIME_LOCAL_KERNELS_READ_H

#include <parser/metadata/MetaDataParser.h>
#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/io/File.h>
#include <runtime/local/io/ReadCsv.h>
#include <runtime/local/io/ReadDaphne.h>
#include <runtime/local/io/ReadMM.h>
#include <runtime/local/io/ReadParquet.h>
#if USE_HDFS
#include <runtime/local/io/HDFS/ReadHDFS.h>
#endif

#include <filesystem>
#include <string>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes> struct Read {
    static void apply(DTRes *&res, const char *filename, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes> void read(DTRes *&res, const char *filename, DCTX(ctx)) {
    Read<DTRes>::apply(res, filename, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct Read<DenseMatrix<VT>> {
    static void apply(DenseMatrix<VT> *&res, const char *filename, DCTX(ctx)) {
        FileMetaData fmd = MetaDataParser::readMetaData(filename);
        std::string ext(std::filesystem::path(filename).extension());

        if (ext == ".csv") {
            if (res == nullptr)
                res = DataObjectFactory::create<DenseMatrix<VT>>(fmd.numRows, fmd.numCols, false);
            readCsv(res, filename, fmd.numRows, fmd.numCols, ',');
        } else if (ext == ".mtx") {
            if constexpr (std::is_same<VT, std::string>::value)
                throw std::runtime_error("reading string-valued MatrixMarket files is not supported (yet)");
            else
                readMM(res, filename);
        } else if (ext == ".parquet") {
            if constexpr (std::is_same<VT, std::string>::value)
                throw std::runtime_error("reading string-valued Parquet files is not supported (yet)");
            else {
                if (res == nullptr)
                    res = DataObjectFactory::create<DenseMatrix<VT>>(fmd.numRows, fmd.numCols, false);
                readParquet(res, filename, fmd.numRows, fmd.numCols);
            }
        } else if (ext == ".dbdf") {
            if constexpr (std::is_same<VT, std::string>::value)
                throw std::runtime_error("reading string-valued DAPHNE binary format files is not supported (yet)");
            else
                readDaphne(res, filename);
        }
#if USE_HDFS
        else if (ext == ".hdfs") {
            if constexpr (std::is_same<VT, std::string>::value)
                throw std::runtime_error("reading string-valued HDFS files is not supported (yet)");
            else {
                if (res == nullptr)
                    res = DataObjectFactory::create<DenseMatrix<VT>>(fmd.numRows, fmd.numCols, false);
                readHDFS(res, filename, ctx);
            }
        }
#endif
        else
            throw std::runtime_error("file extension not supported: '" + ext + "'");
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct Read<CSRMatrix<VT>> {
    static void apply(CSRMatrix<VT> *&res, const char *filename, DCTX(ctx)) {
        FileMetaData fmd = MetaDataParser::readMetaData(filename);
        std::string ext(std::filesystem::path(filename).extension());

        if (ext == ".csv") {
            if (fmd.numNonZeros == -1)
                throw std::runtime_error("currently reading of sparse matrices requires a number of "
                                         "non zeros to be defined");

            if (res == nullptr)
                res = DataObjectFactory::create<CSRMatrix<VT>>(fmd.numRows, fmd.numCols, fmd.numNonZeros, false);

            // FIXME: ensure file is sorted, or set `sorted` argument correctly
            readCsv(res, filename, fmd.numRows, fmd.numCols, ',', fmd.numNonZeros, true);
        } else if (ext == ".mtx") {
            readMM(res, filename);
        } else if (ext == ".parquet") {
            if (res == nullptr)
                res = DataObjectFactory::create<CSRMatrix<VT>>(fmd.numRows, fmd.numCols, fmd.numNonZeros, false);
            readParquet(res, filename, fmd.numRows, fmd.numCols, fmd.numNonZeros, false);
        } else if (ext == ".dbdf")
            readDaphne(res, filename);
        else
            throw std::runtime_error("file extension not supported: '" + ext + "'");
    }
};

// ----------------------------------------------------------------------------
// Frame
// ----------------------------------------------------------------------------

template <> struct Read<Frame> {
    static void apply(Frame *&res, const char *filename, DCTX(ctx)) {
        FileMetaData fmd = MetaDataParser::readMetaData(filename);
        std::string ext(std::filesystem::path(filename).extension());

        if (ext == ".csv") {
            ValueTypeCode *schema;
            if (fmd.isSingleValueType) {
                schema = new ValueTypeCode[fmd.numCols];
                for (size_t i = 0; i < fmd.numCols; i++)
                    schema[i] = fmd.schema[0];
            } else
                schema = fmd.schema.data();

            std::string *labels;
            if (fmd.labels.empty())
                labels = nullptr;
            else
                labels = fmd.labels.data();

            if (res == nullptr)
                res = DataObjectFactory::create<Frame>(fmd.numRows, fmd.numCols, schema, labels, false);

            readCsv(res, filename, fmd.numRows, fmd.numCols, ',', schema);

            if (fmd.isSingleValueType)
                delete[] schema;
        } else
            throw std::runtime_error("file extension not supported: '" + ext + "'");
    }
};

#endif // SRC_RUNTIME_LOCAL_KERNELS_READ_H
