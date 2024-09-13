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

#include <parser/metadata/MetaDataParser.h>

#if USE_HDFS

#include <hdfs/hdfs.h>

#include <filesystem>

struct HDFSUtils {
    static FileMetaData parseHDFSMetaData(std::string filename, const hdfsFS &fs) {
        auto metaFn = (filename + ".meta");
        // Read related metadata in hdfs
        hdfsFile hFile = hdfsOpenFile(fs, metaFn.c_str(), O_RDONLY, 0, 0, 0);
        hdfsFileInfo *fileInfo = hdfsGetPathInfo(fs, metaFn.c_str());
        size_t fileSize = fileInfo->mSize;
        hdfsFreeFileInfo(fileInfo, 1);
        std::vector<char> buffer(fileSize);
        hdfsRead(fs, hFile, buffer.data(), fileSize);
        std::string fmdStr(buffer.data(), buffer.size());
        return MetaDataParser::readMetaDataFromString(fmdStr);
    }

    static inline std::string getBaseFile(const char * fn){
        std::filesystem::path filePath(fn);
        return filePath.filename().string();
    }

    static std::tuple<int, size_t> findSegmendAndOffset(const hdfsFS &fs, size_t startOffset, size_t startRow, const char * filename, size_t offsetMuliplier) {
        size_t skippedRows = 0;
        int seg = 1;
        
        while (skippedRows != startRow) {
            std::string segf = std::string(filename) + "/" + HDFSUtils::getBaseFile(filename) + "_segment_" + std::to_string(seg++);
            auto fmd = HDFSUtils::parseHDFSMetaData(segf, fs);
            skippedRows += fmd.numRows;
            // We need offset within segment
            if (skippedRows > startRow) {
                seg--;  // adjust segment and skipped rows
                skippedRows -= fmd.numRows;
                startOffset += (startRow - skippedRows) * offsetMuliplier;
                skippedRows = startRow;
            }
        }
        return std::make_tuple(seg, startOffset);
    }

    static std::tuple<std::string, uint16_t> parseIPAddress(const std::string& input) {
        std::string ip;
        uint16_t port = 9000;  // Default port
        size_t colonPos = input.find(':');

        if (colonPos != std::string::npos) {
            // If there's a colon, split IP and port
            ip = input.substr(0, colonPos);
            std::string portStr = input.substr(colonPos + 1);
            port = static_cast<uint16_t>(std::stoi(portStr));
        } else {
            // If no colon, the entire input is the IP
            ip = input;
        }

        return std::make_tuple(ip, port);
    }
};
#endif