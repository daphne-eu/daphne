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
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/io/File.h>
#include <runtime/local/io/utils.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <type_traits>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTArg> struct WriteHDFSCsv {
    static void apply(const DTArg *arg, const char *hdfsFilename, DCTX(dctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTArg> void writeHDFSCsv(const DTArg *arg, const char *hdfsFilename, DCTX(dctx)) {
    WriteHDFSCsv<DTArg>::apply(arg, hdfsFilename, dctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct WriteHDFSCsv<DenseMatrix<VT>> {
    static void apply(const DenseMatrix<VT> *arg, const char *hdfsFilename, DCTX(dctx)) {
        if (hdfsFilename == nullptr) {
            throw std::runtime_error("Could not read hdfs file");
        }

        std::string fn(hdfsFilename);

        auto hdfsCtx = HDFSContext::get(dctx);
        auto fs = hdfsCtx->getConnection();
        if (fs == NULL) {
            std::cout << "Error connecting to HDFS" << std::endl;
        }

        // Check if path exists
        std::filesystem::path filePath(hdfsFilename);
        auto dirFileName = filePath.parent_path();

        if (hdfsExists(*fs, dirFileName.c_str()) == -1) {
            // The file does not exist, so create the directory structure
            // and the file
            // TODO extract directory path from filename
            if (fn.find("/") == std::string::npos)
                throw std::runtime_error("HDFS subdirectories not supported atm");
            // if(hdfsCreateDirectory(fs, fn.c_str()) == -1)
            //     throw std::runtime_error("Failed to create file");
        }
        // If not, add "/" at the beginning
        if (fn.find("/") == std::string::npos)
            fn = "/" + fn;

        // Write related fmd
        FileMetaData fmd(arg->getNumRows(), arg->getNumCols(), true, ValueTypeUtils::codeFor<VT>);
        auto fmdStr = MetaDataParser::writeMetaDataToString(fmd);
        auto mdtFn = fn + ".meta";
        hdfsFile hdfsFile = hdfsOpenFile(*fs, mdtFn.c_str(), O_WRONLY, 0, 0, 0);
        if (hdfsFile == NULL) {
            throw std::runtime_error("Error opening HDFS file");
        }
        hdfsWrite(*fs, hdfsFile, static_cast<const void *>(fmdStr.c_str()), fmdStr.size());
        hdfsCloseFile(*fs, hdfsFile);
        // Open the HDFS file for writing
        hdfsFile = hdfsOpenFile(*fs, fn.c_str(), O_WRONLY, 0, 0, 0);
        if (hdfsFile == NULL) {
            throw std::runtime_error("Error opening HDFS file");
        }

        const VT *valuesArg = arg->getValues();
        size_t cell = 0;
        for (size_t i = 0; i < arg->getNumRows(); ++i) {
            for (size_t j = 0; j < arg->getNumCols(); ++j) {
                // Convert the numeric value to a string and add a comma
                std::ostringstream oss;
                oss << valuesArg[cell];
                oss << (j < (arg->getNumCols() - 1) ? "," : ""); // Add a comma unless it's the last column
                const std::string &valueStr = oss.str();

                // Write the value string to the HDFS file
                if (hdfsWrite(*fs, hdfsFile, static_cast<const void *>(valueStr.c_str()), valueStr.size()) == -1) {
                    hdfsCloseFile(*fs, hdfsFile);
                    throw std::runtime_error("Failed to write to HDFS file");
                }
                cell++;
            }
            // Add a newline character at the end of each row
            const char newline = '\n';
            if (hdfsWrite(*fs, hdfsFile, static_cast<const void *>(&newline), 1) == -1) {
                hdfsCloseFile(*fs, hdfsFile);
                throw std::runtime_error("Failed to write to HDFS file");
            }
        }

        // Close the HDFS file and disconnect
        if (hdfsCloseFile(*fs, hdfsFile) == -1) {
            throw std::runtime_error("Failed to close HDFS file");
        }
    }
};
