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

#include <runtime/local/io/File.h>
#include <runtime/local/io/utils.h>
#include <runtime/local/io/DaphneSerializer.h>
#include <runtime/local/io/HDFS/WriteHDFSCsv.h>
#include <runtime/local/io/HDFS/WriteDaphneHDFS.h>
#include <runtime/distributed/coordinator/kernels/DistributedWrite.h>
#include <runtime/local/datastructures/AllocationDescriptorGRPC.h>

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
struct WriteHDFS
{
    static void apply(const DTArg *arg, const char *filename, DCTX(dctx)) 
    {
        auto hdfsCtx = HDFSContext::get(dctx);
        auto fs = hdfsCtx->getConnection();
        if (fs == NULL)
        {
            std::cout << "Error connecting to HDFS" << std::endl;
        }

        // TODO for now store to /filename/ directory in hdfs.
        std::filesystem::path filePath(filename);
        auto dirFileName = "/" + filePath.filename().string();
        auto baseFileName = dirFileName + "/" + filePath.filename().string();

        // Get nested file extension
        auto extension = filePath.stem().extension().string();

        // Check if the directory already exists
        if (hdfsExists(*fs, dirFileName.c_str()) == -1) {
            // The file does not exist, so create the directory structure
            // and the file
            if(hdfsCreateDirectory(*fs, dirFileName.c_str()) == -1)
                throw std::runtime_error("Failed to create file");
        } else {
            // clear directory
            int numEntries;
            hdfsFileInfo * files = hdfsListDirectory(*fs, dirFileName.c_str(), &numEntries);
            for (int i = 0; i < numEntries; i++)
                hdfsDelete(*fs, files[i].mName, 1);
            hdfsFreeFileInfo(files, numEntries);
        }

        if (dctx->config.use_distributed) {
            distributedWrite<DTArg>(arg, filename, dctx);
           
        } else {
            // Write one segment
            auto hdfsfilename = baseFileName + "_segment_1";            
            if (extension == ".csv") {                
                writeHDFSCsv(arg, hdfsfilename.c_str(), dctx);
            } else if (extension == ".dbdf") {
                writeDaphneHDFS(arg, hdfsfilename.c_str(), dctx);
            }
        }  
    }
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTArg>
void writeHDFS(const DTArg *arg, const char *filename, DCTX(dctx))
{
    WriteHDFS<DTArg>::apply(arg, filename, dctx);
}
