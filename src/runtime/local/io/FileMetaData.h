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

#ifndef SRC_RUNTIME_LOCAL_IO_FILEMETADATA_H
#define SRC_RUNTIME_LOCAL_IO_FILEMETADATA_H

#include <runtime/local/datastructures/ValueTypeCode.h>

#include <fstream>
#include <string>
#include <vector>

#include <cstddef>
#include <cstdlib>
#include <cstring>

/**
 * @brief Very simple representation of basic file meta data.
 * 
 * Currently tailored to frames.
 */
struct FileMetaData {
    const size_t numRows;
    const size_t numCols;
    bool isSingleValueType;
    std::vector<ValueTypeCode> schema;
    std::vector<std::string> labels;
    
    FileMetaData(
            size_t numRows,
            size_t numCols,
            bool isSingleValueType,
            std::vector<ValueTypeCode> schema,
            std::vector<std::string> labels
    ) :
            numRows(numRows), numCols(numCols),
            isSingleValueType(isSingleValueType), schema(schema),
            labels(labels)
    {
        //
    }
    
    /**
     * @brief Retrieves the file meta data for the specified file.
     * 
     * @param filename The name of the file for which to retrieve the meta
     * data. Note that the extension ".meta" is appended to this filename to
     * determine the name of the meta data file.
     * @return The meta data of the specified file.
     */
    static FileMetaData ofFile(const std::string filename) {
        std::ifstream ifs(filename + ".meta", std::ios::in);
        if (!ifs.good())
            throw std::runtime_error(
                    "could not open file '" + filename +
                    "' for reading meta data"
            );

        const size_t bufSize = 1024;
        char buf[bufSize];

        ifs.getline(buf, bufSize, ',');
        const size_t numRows = atoll(buf);
        
        ifs.getline(buf, bufSize, ',');
        const size_t numCols = atoll(buf);
        
        ifs.getline(buf, bufSize, ',');
        const bool isSingleValueType = atoi(buf);

        std::vector<ValueTypeCode> schema;
        const size_t expectedNumColTypes = isSingleValueType ? 1 : numCols;
        for(size_t i = 0; i < expectedNumColTypes; i++) {
            ifs.getline(buf, bufSize, ',');
            ValueTypeCode vtc;
                 if(!strncmp(buf, "f64" , bufSize)) vtc = ValueTypeCode::F64;
            else if(!strncmp(buf, "f32" , bufSize)) vtc = ValueTypeCode::F32;
            else if(!strncmp(buf, "si64", bufSize)) vtc = ValueTypeCode::SI64;
            else if(!strncmp(buf, "si32", bufSize)) vtc = ValueTypeCode::SI32;
            else if(!strncmp(buf, "si8" , bufSize)) vtc = ValueTypeCode::SI8;
            else if(!strncmp(buf, "ui64", bufSize)) vtc = ValueTypeCode::UI64;
            else if(!strncmp(buf, "ui32", bufSize)) vtc = ValueTypeCode::UI32;
            else if(!strncmp(buf, "ui8" , bufSize)) vtc = ValueTypeCode::UI8;
            else
                throw std::runtime_error(
                        std::string("unknown value type: ") + buf
                );
            schema.push_back(vtc);
        }
        
        std::vector<std::string> labels;
        
        // Labels are optional.
        const std::streamoff pos = ifs.tellg();
        ifs.seekg(0, std::ios_base::end);
        if(pos != ifs.tellg()) {
            // If we have not reached the end of the stream yet.
            ifs.seekg(pos, std::ios_base::beg);
            for(size_t i = 0; i < numCols; i++) {
                ifs.getline(buf, bufSize, ',');
                labels.push_back(buf);
            }
        }
        // else: labels remains empty

        return FileMetaData(numRows, numCols, isSingleValueType, schema, labels);
    }
};

#endif //SRC_RUNTIME_LOCAL_IO_FILEMETADATA_H