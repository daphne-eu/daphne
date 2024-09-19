/*
 * Copyright 2023 The DAPHNE Consortium
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
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <runtime/local/io/File.h>
#include <runtime/local/io/utils.h>

#include <runtime/local/io/HDFS/HDFSUtils.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes> struct ReadHDFSCsv {
    static void apply(DTRes *&res, const char *hdfsDir, size_t numRows,
                      size_t numCols, char delim, DCTX(dctx),
                      size_t startRow = 0) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes>
void readHDFSCsv(DTRes *&res, const char *hdfsDir, size_t numRows,
                 size_t numCols, char delim, DCTX(dctx), size_t startRow = 0) {
    ReadHDFSCsv<DTRes>::apply(res, hdfsDir, numRows, numCols, delim, dctx,
                              startRow);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct ReadHDFSCsv<DenseMatrix<VT>> {
    static void apply(DenseMatrix<VT> *&res, const char *hdfsDir,
                      size_t numRows, size_t numCols, char delim, DCTX(dctx),
                      size_t startRow = 0) {
        if (hdfsDir == nullptr) {
            throw std::runtime_error("File required");
        }
        if (numRows <= 0) {
            throw std::runtime_error("numRows must be > 0");
        }
        if (numRows <= 0) {
            throw std::runtime_error("numCols must be > 0");
        }

        if (res == nullptr) {
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols,
                                                             false);
        }

        auto hdfsCtx = HDFSContext::get(dctx);
        auto fs = hdfsCtx->getConnection();
        if (fs == NULL) {
            std::cerr << "Error connecting to HDFS" << std::endl;
        }

        [[maybe_unused]] auto [startSegment, dummy] =
            HDFSUtils::findSegmendAndOffset(*fs, 0, startRow, hdfsDir,
                                            numCols * sizeof(VT));
        // TODO verify file exists

        size_t parsedRows = 0;
        auto segment = startSegment;
        if (res == NULL) {
            throw std::runtime_error("Could not initialize result matrix");
        }

        VT *valuesRes = res->getValues();

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

            char buffer[1UL << 20];
            char *cur = nullptr;
            size_t n = 0;

            for (size_t r = 0; r < segFmd.numRows; r++) {
                std::string line;

                do {
                    if (cur == nullptr) {
                        n = hdfsRead(*fs, hFile, buffer, 1UL << 20);
                        if (n <= 0) {
                            throw std::runtime_error(
                                "Could not read hdfs file");
                        }
                        cur = buffer;
                    }

                    char *eol = (char *)std::memchr(cur, '\n', n);
                    if (eol == nullptr || static_cast<size_t>(eol - cur) >= n) {
                        line.append(cur, n);
                        cur = nullptr;
                    } else {
                        line.append(cur, eol - cur);
                        cur = eol + 1;
                    }
                } while (cur == nullptr);
                // If first segment, skip rows
                if (parsedRows == 0 &&
                    startRow > (segment - 2) * segFmd.numRows + r)
                    continue;

                size_t pos = 0;
                for (size_t c = 0; c < numCols; c++) {
                    VT val;
                    convertCstr(line.c_str() + pos, &val);

                    // TODO This assumes that rowSkip == numCols.
                    *valuesRes = val;
                    valuesRes++;
                    // TODO We could even exploit the fact that the strtoX
                    // functions can return a pointer to the first character
                    // after the parsed input, then we wouldn't have to search
                    // for that ourselves, just would need to check if it is
                    // really the delimiter.
                    if (c < numCols - 1) {
                        while (line[pos] != delim)
                            pos++;
                        pos++; // skip delimiter
                    }
                }
                parsedRows++;
                if (parsedRows == numRows)
                    break;
            }

            hdfsCloseFile(*fs, hFile);
        }
    }
};
