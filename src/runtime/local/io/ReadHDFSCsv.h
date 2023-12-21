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

#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <runtime/local/io/File.h>
#include <runtime/local/io/utils.h>

#include <cassert>

#include <hdfs.h>


// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes> struct ReadHDFSCsv {
  static void apply(DTRes *&res, const char *filename, size_t numRows, size_t numCols,
                    char delim, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes>
void readHDFSCsv(DTRes *&res, const char *filename, size_t numRows, size_t numCols,
             char delim, DCTX(ctx)) {
  ReadHDFSCsv<DTRes>::apply(res, filename, numRows, numCols, delim, ctx);
}


// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct ReadHDFSCsv<DenseMatrix<VT>> {
  static void apply(DenseMatrix<VT> *&res, const char *filename, size_t numRows,
                    size_t numCols, char delim, DCTX(ctx)) {
    assert(filename != nullptr && "File required");
    assert(numRows > 0 && "numRows must be > 0");
    assert(numCols > 0 && "numCols must be > 0");

    const char *host = "hdfs://10.0.1.94";
    const char *user = "ubuntu";
    tPort port = 9000;
    
    hdfsFS fs = hdfsConnectAsUser(host, port, user);
    if (fs == NULL) {
        std::cerr << "Error connecting to HDFS" << std::endl;
    }

    hdfsFile hFile = hdfsOpenFile(fs, filename, O_RDONLY, 0, 0, 0);
    if (hFile == NULL) {
        std::cerr << "Error opening HDFS file" << std::endl;
        hdfsDisconnect(fs);
    }

    int nb;
    BlockLocation *bls = hdfsGetFileBlockLocations(fs, filename, 0, 1UL << 20, &nb);
    for (int i = 0; i < nb; i++) {
        std::cerr << "block: " << i << std::endl;

	BlockLocation bl = bls[i];

	for (int j = 0; j < bl.numOfNodes; j++) {
            std::cerr << "host: " << bl.hosts[j] << std::endl;
            std::cerr << "topo: " << bl.topologyPaths[j] << std::endl;
	}
    }

    if (res == nullptr) {
        res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false);
    }
    assert(res != NULL && "Could not initialize result matrix");
    VT *valuesRes = res->getValues();

    char buffer[1UL << 20];
    char *cur = nullptr;
    size_t n = 0;

    for (size_t r = 0; r < numRows; r++) {
        std::string line;

        do {
            if (cur == nullptr) {
                n = hdfsRead(fs, hFile, buffer, 1UL << 20);
                assert(n > 0 && "Could not read hdfs file");
                cur = buffer;
            }

            char *eol = (char *)std::memchr(cur, '\n', n);
            if (eol == nullptr || eol - cur >= n) {
                line.append(cur, n);
                cur = nullptr;
            } else {
                line.append(cur, eol - cur);
                cur = eol + 1;
            }
        } while (cur == nullptr);
        std::cout << line << std::endl;
        // char *next = (char *)line.c_str();
        // for (size_t c = 0; c < numCols; c++, valuesRes++, next++) {
        //     convertCstr(next, valuesRes, &next);
        // }
    }

    hdfsCloseFile(fs, hFile);
    hdfsDisconnect(fs);

    res->print(std::cerr);
  }
};
