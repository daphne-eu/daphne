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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_WRITE_H
#define SRC_RUNTIME_LOCAL_KERNELS_WRITE_H

#include <parser/metadata/MetaDataParser.h>
#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Matrix.h>
#include <runtime/local/io/File.h>
#include <runtime/local/io/FileMetaData.h>
#include <runtime/local/io/WriteCsv.h>
#include <runtime/local/io/WriteDaphne.h>
#if USE_HDFS
#include <runtime/local/io/HDFS/WriteHDFS.h>
#endif

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTArg> struct Write {
    static void apply(const DTArg *arg, const char *filename,
                      DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTArg>
void write(const DTArg *arg, const char *filename, DCTX(ctx)) {
    Write<DTArg>::apply(arg, filename, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct Write<DenseMatrix<VT>> {
    static void apply(const DenseMatrix<VT> *arg, const char *filename,
                      DCTX(ctx)) {
        std::string fn(filename);
        auto pos = fn.find_last_of('.');
        std::string ext(fn.substr(pos + 1));
        if (ext == "csv") {
            File *file = openFileForWrite(filename);
            FileMetaData metaData(arg->getNumRows(), arg->getNumCols(), true,
                                  ValueTypeUtils::codeFor<VT>);
            MetaDataParser::writeMetaData(filename, metaData);
            writeCsv(arg, file);
            closeFile(file);
        } else if (ext == "dbdf") {
            FileMetaData metaData(arg->getNumRows(), arg->getNumCols(), true,
                                  ValueTypeUtils::codeFor<VT>);
            MetaDataParser::writeMetaData(filename, metaData);
            writeDaphne(arg, filename);
#if USE_HDFS
        } else if (ext == "hdfs") {
            HDFSMetaData hdfs = {true, filename};
            FileMetaData metaData(arg->getNumRows(), arg->getNumCols(), true,
                                  ValueTypeUtils::codeFor<VT>, -1, hdfs);
            // Get file extension before .hdfs (e.g. file.csv.hdfs)
            auto posHdfs = pos;
            auto posExt = fn.find_last_of('.', pos - 1);
            std::string nestedExt(fn.substr(posExt + 1, posHdfs - posExt - 1));
            MetaDataParser::writeMetaData(filename, metaData);

            // call WriteHDFS
            writeHDFS(arg, filename, ctx);
#endif
        }
    }
};

// ----------------------------------------------------------------------------
// Frame
// ----------------------------------------------------------------------------

template <> struct Write<Frame> {
    static void apply(const Frame *arg, const char *filename, DCTX(ctx)) {
        File *file = openFileForWrite(filename);
        std::vector<ValueTypeCode> vtcs;
        std::vector<std::string> labels;
        for (size_t i = 0; i < arg->getNumCols(); i++) {
            vtcs.push_back(arg->getSchema()[i]);
            labels.push_back(arg->getLabels()[i]);
        }
        FileMetaData metaData(arg->getNumRows(), arg->getNumCols(), false, vtcs,
                              labels);
        MetaDataParser::writeMetaData(filename, metaData);
        writeCsv(arg, file);
        closeFile(file);
    }
};

// ----------------------------------------------------------------------------
// Matrix
// ----------------------------------------------------------------------------

template <typename VT> struct Write<Matrix<VT>> {
    static void apply(const Matrix<VT> *arg, const char *filename, DCTX(ctx)) {
        std::string fn(filename);
        auto pos = fn.find_last_of('.');
        std::string ext(fn.substr(pos + 1));
        if (ext == "csv") {
            File *file = openFileForWrite(filename);
            FileMetaData metaData(arg->getNumRows(), arg->getNumCols(), true,
                                  ValueTypeUtils::codeFor<VT>);
            MetaDataParser::writeMetaData(filename, metaData);
            writeCsv(arg, file);
            closeFile(file);
        } else {
            throw std::runtime_error(
                "[Write.h] - generic Matrix type currently only supports csv "
                "file extension.");
        }
    }
};

#endif // SRC_RUNTIME_LOCAL_KERNELS_WRITE_H
