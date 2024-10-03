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
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>

#include <parser/metadata/MetaDataParser.h>
#include <runtime/distributed/coordinator/kernels/DistributedRead.h>
#include <runtime/local/io/DaphneSerializer.h>
#include <runtime/local/io/File.h>
#include <runtime/local/io/HDFS/ReadDaphneHDFS.h>
#include <runtime/local/io/HDFS/ReadHDFSCsv.h>
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

template <class DTRes> struct ReadHDFS {
    static void apply(DTRes *&res, const char *filename, DCTX(dctx)) {
        FileMetaData fmd = MetaDataParser::readMetaData(filename);
        res = DataObjectFactory::create<DTRes>(fmd.numRows, fmd.numCols, false);

        auto hdfsFilename = fmd.hdfs.HDFSFilename.c_str();
        std::filesystem::path filePath(hdfsFilename);

        // Get nested file extension
        auto extension = filePath.stem().extension().string();

        if (dctx->config.use_distributed) {
            distributedRead<DTRes>(res, hdfsFilename, dctx);
        } else {
            if (extension == ".csv") {
                readHDFSCsv(res, hdfsFilename, fmd.numRows, fmd.numCols, ',', dctx);
            } else if (extension == ".dbdf") {
                readDaphneHDFS(res, hdfsFilename, dctx);
            }
        }
    }
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes> void readHDFS(DTRes *&res, const char *hdfsFilename, DCTX(dctx)) {
    ReadHDFS<DTRes>::apply(res, hdfsFilename, dctx);
}
