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

#ifndef SRC_RUNTIME_LOCAL_IO_WRITEHDFSCSV_H
#define SRC_RUNTIME_LOCAL_IO_WRITEHDFSCSV_H

#include <hdfs.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/io/File.h>
#include <runtime/local/io/utils.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <type_traits>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTArg>
struct WriteHDFSCsv {
    static void apply(const DTArg *arg, const char *hdfsFilename) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTArg>
void writeHDFSCsv(const DTArg *arg, const char *hdfsFilename) {
    WriteHDFSCsv<DTArg>::apply(arg, hdfsFilename);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT>
struct WriteHDFSCsv<DenseMatrix<VT>> {
    static void apply(const DenseMatrix<VT> *arg, const char *hdfsFilename) {
        assert(hdfsFilename != nullptr && "File path required");

        std::string fn(hdfsFilename);
        const char *host = "hdfs://10.0.1.94";
        tPort port = 9000;  // Default port
        const char *user = "ubuntu";

        hdfsFS fs = hdfsConnectAsUser(host, port, user);
        if (fs == NULL) {
            std::cout << "Error connecting to HDFS" << std::endl;
        }

        // Check if the file already exists
        if (hdfsExists(fs, fn.c_str()) == -1) {
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

        // Open the HDFS file for reading
        hdfsFile hdfsFile = hdfsOpenFile(fs, fn.c_str(), O_WRONLY, 0, 0, 0);
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
                oss << (j < (arg->getNumCols() - 1)
                            ? ","
                            : "");  // Add a comma unless it's the last column
                const std::string &valueStr = oss.str();

                // Write the value string to the HDFS file
                if (hdfsWrite(fs, hdfsFile,
                              static_cast<const void *>(valueStr.c_str()),
                              valueStr.size()) == -1) {
                    hdfsCloseFile(fs, hdfsFile);
                    hdfsDisconnect(fs);
                    throw std::runtime_error("Failed to write to HDFS file");
                }
                cell++;
            }
            // Add a newline character at the end of each row
            const char newline = '\n';
            if (hdfsWrite(fs, hdfsFile, static_cast<const void *>(&newline),
                          1) == -1) {
                hdfsCloseFile(fs, hdfsFile);
                hdfsDisconnect(fs);
                throw std::runtime_error("Failed to write to HDFS file");
            }
        }

        // Close the HDFS file and disconnect
        if (hdfsCloseFile(fs, hdfsFile) == -1) {
            throw std::runtime_error("Failed to close HDFS file");
        }

        hdfsDisconnect(fs);
    }
};

#endif  // SRC_RUNTIME_LOCAL_IO_WRITEHDFSCSV_H
