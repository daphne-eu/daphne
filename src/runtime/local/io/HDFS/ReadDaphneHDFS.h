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

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/context/HDFSContext.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>

#include <runtime/local/io/DaphneSerializer.h>
#include <runtime/local/io/File.h>
#include <runtime/local/io/utils.h>

#include <util/preprocessor_defs.h>

#include <type_traits>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip> // For setfill and setw
#include <iostream>
#include <limits>
#include <queue>
#include <sstream>
#include <vector>

#include <fstream>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes> struct ReadDaphneHDFS {
    static void apply(DTRes *&res, const char *hdfsDir, DCTX(dctx),
                      size_t startRow = 0) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes>
void readDaphneHDFS(DTRes *&res, const char *hdfsDir, DCTX(dctx),
                    size_t startRow = 0) {
    ReadDaphneHDFS<DTRes>::apply(res, hdfsDir, dctx, startRow);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct ReadDaphneHDFS<DenseMatrix<VT>> {
    static void apply(DenseMatrix<VT> *&res, const char *hdfsDir, DCTX(dctx),
                      size_t startRow = 0) {
        if (res == NULL) {
            throw std::runtime_error("Could not initialize result matrix");
        }

        size_t numRows = res->getNumRows();
        size_t numCols = res->getNumCols();

        auto hdfsCtx = HDFSContext::get(dctx);
        auto fs = hdfsCtx->getConnection();
        if (fs == NULL) {
            std::cout << "Error connecting to HDFS" << std::endl;
        }

        auto headerSize = DaphneSerializer<DenseMatrix<VT>>::headerSize(res);

        auto [startSegment, offset] = HDFSUtils::findSegmendAndOffset(
            *fs, headerSize, startRow, hdfsDir, numCols * sizeof(VT));

        size_t parsedRows = 0;
        auto segment = startSegment;
        size_t startSerByte = headerSize;
        while (parsedRows < numRows) {
            auto hdfsFn = std::string(hdfsDir) + "/" +
                          HDFSUtils::getBaseFile(hdfsDir) + "_segment_" +
                          std::to_string(segment++);
            auto segFmd = HDFSUtils::parseHDFSMetaData(hdfsFn, *fs);

            hdfsFile hFile =
                hdfsOpenFile(*fs, hdfsFn.c_str(), O_RDONLY, 0, 0, 0);
            if (hFile == NULL) {
                throw std::runtime_error("Error opening HDFS file");
            }

            // Find out the size of the file to allocate a buffer
            hdfsFileInfo *fileInfo = hdfsGetPathInfo(*fs, hdfsFn.c_str());
            if (fileInfo == NULL) {
                hdfsCloseFile(*fs, hFile);
                throw std::runtime_error("Error getting file info");
                return;
            }
            tSize fileSize = fileInfo->mSize;
            hdfsFreeFileInfo(fileInfo, 1);

            // Allocate buffer
            std::vector<char> buffer(fileSize);

            // If started parsing rows, set offset to headerSize
            // Simply skip buffer header, except first time.
            offset = parsedRows == 0 ? offset : headerSize;
            hdfsSeek(*fs, hFile, offset);
            // Read the file into the buffer
            tSize bytesRead = hdfsRead(*fs, hFile, buffer.data(), fileSize);
            if (bytesRead == -1) {
                hdfsCloseFile(*fs, hFile);
                throw std::runtime_error("Error reading file");
                return;
            }
            size_t bufferEnd = fileSize;
            // If segment is bigger than end row, bufferend should not be equal
            // to whole segment
            if (numRows - parsedRows < segFmd.numRows) {
                bufferEnd = (numRows - parsedRows) * numCols * sizeof(VT);
            }
            res = DaphneSerializer<DenseMatrix<VT>>::deserialize(
                buffer.data(), bufferEnd, res, startSerByte);
            startSerByte += bufferEnd - offset;

            hdfsCloseFile(*fs, hFile);
            parsedRows += segFmd.numRows;
        }
    }
};
