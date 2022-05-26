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
    const ssize_t numNonZeros;
    
    FileMetaData(
            size_t numRows,
            size_t numCols,
            bool isSingleValueType,
            std::vector<ValueTypeCode> schema,
            std::vector<std::string> labels,
            ssize_t numNonZeros = -1
    ) :
            numRows(numRows), numCols(numCols),
            isSingleValueType(isSingleValueType), schema(schema),
            labels(labels), numNonZeros(numNonZeros)
    {
        //
    }
    
    /**
     * @deprecated Since JSON parser for meta data has been added.
     */
    static void toFile(const std::string filename, size_t numRows, size_t numCols, bool isSingleValueType, ValueTypeCode vtc)
    {
        std::string vtc_;
             if(vtc == ValueTypeCode::F64)  vtc_ = "f64";
        else if(vtc == ValueTypeCode::F32)  vtc_ = "f32";
        else if(vtc == ValueTypeCode::SI64) vtc_ = "si64";
        else if(vtc == ValueTypeCode::SI32) vtc_ = "si32";
        else if(vtc == ValueTypeCode::SI8)  vtc_ = "si8";
        else if(vtc == ValueTypeCode::UI64) vtc_ = "ui64";
        else if(vtc == ValueTypeCode::UI32) vtc_ = "ui32";
        else if(vtc == ValueTypeCode::UI8)  vtc_ = "ui8";
        else throw std::runtime_error("FileMetaData::toFile: unknown value type code");
        std::ofstream ofs(filename + ".meta", std::ios::out);
        if (!ofs.good())
            throw std::runtime_error(
                    "could not open file '" + filename +
                    "' for writing meta data"
            );
        if(ofs.is_open())
            ofs << numRows << "," << numCols << "," << isSingleValueType << "," << vtc_;
    }
    
    /** 
     * @brief Retrieves the file meta data for the specified file.
     * 
     * @deprecated Since JSON parser for meta data has been added.
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

        // TODO: number of non zeros should not require labels to be set (improve meta format)
        ssize_t numNonZeros = -1;
        // Labels are optional.
        if(!ifs.eof()) {
            ifs.getline(buf, bufSize, ',');
            std::string optLine(buf);
            size_t i = 0;
            if(optLine.find("nnz=") == 0) {
                numNonZeros = std::stoll(optLine.substr(std::strlen("nnz=")));
            }
            else {
                // If we have not reached the end of the stream yet.
                labels.emplace_back(buf);
                i++;
            }
            if(!ifs.eof()) {
                for(; i < numCols ; i++) {
                    ifs.getline(buf, bufSize, ',');
                    labels.emplace_back(buf);
                }
            }
        }
        // else: labels remains empty

        return FileMetaData(numRows, numCols, isSingleValueType, schema, labels, numNonZeros);
    }
};

#endif //SRC_RUNTIME_LOCAL_IO_FILEMETADATA_H