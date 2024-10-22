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
#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/context/HDFSContext.h>

#include <runtime/local/io/File.h>
#include <runtime/local/io/utils.h>
#include <runtime/local/io/DaphneSerializer.h>

#include <util/preprocessor_defs.h>

#include <type_traits>

#include <cstddef>
#include <cstdint>
#include <queue>
#include <fstream>
#include <limits>
#include <sstream>
#include <iostream>

#include <fstream>
#include <vector>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTArg>
struct WriteDaphneHDFS
{
    static void apply(const DTArg *arg, const char *hdfsFilename, DCTX(dctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTArg>
void writeDaphneHDFS(const DTArg *arg, const char *hdfsFilename, DCTX(dctx))
{
    WriteDaphneHDFS<DTArg>::apply(arg, hdfsFilename, dctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT>
struct WriteDaphneHDFS<DenseMatrix<VT>>{
    static void apply(const DenseMatrix<VT> *arg, const char *hdfsFilename, DCTX(dctx))
    {
        size_t length;
        length = DaphneSerializer<DenseMatrix<VT>>::length(arg);
                
        std::vector<char> buffer(length);
        DaphneSerializer<DenseMatrix<VT>>::serialize(arg, buffer);

        auto hdfsCtx = HDFSContext::get(dctx);
        auto fs = hdfsCtx->getConnection();
        if (fs == NULL)
        {
            std::cout << "Error connecting to HDFS" << std::endl;
        }

        // Write related fmd
        FileMetaData fmd(arg->getNumRows(), arg->getNumCols(), true, ValueTypeUtils::codeFor<VT>);
        auto fmdStr = MetaDataParser::writeMetaDataToString(fmd);
        auto fn = std::string(hdfsFilename);
        auto mdtFn = fn + ".meta";
        hdfsFile hdfsFile = hdfsOpenFile(*fs, mdtFn.c_str(), O_WRONLY, 0, 0, 0);
        if (hdfsFile == NULL) {
            throw std::runtime_error("Error opening HDFS file");
        }
        hdfsWrite(*fs, hdfsFile, static_cast<const void *>(fmdStr.c_str()), fmdStr.size());
        hdfsCloseFile(*fs, hdfsFile);

        // Write binary
        hdfsFile = hdfsOpenFile(*fs, hdfsFilename, O_WRONLY, 0, 0, 0);
        if (hdfsFile == NULL)
        {
            throw std::runtime_error("Error opening HDFS file");
        }

        hdfsWrite(*fs, hdfsFile, buffer.data(), length);
        if (hdfsCloseFile(*fs, hdfsFile) == -1)
        {
            throw std::runtime_error("Failed to close HDFS file");
        }        
    }
};
