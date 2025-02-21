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

#include <iostream>
#include <runtime/local/io/utils.h>

// create positional map based on csv data

// Function save the positional map
// build a contiguous buffer then write in one shot.
void writePositionalMap(const char* filename, 
                        const std::vector<std::pair<std::streampos, std::vector<std::uint16_t>>>& posMap) {
    
    using clock = std::chrono::high_resolution_clock;
    auto time = clock::now();

    std::string posMapFile = getPosMapFile(filename);
    std::ofstream ofs(posMapFile, std::ios::binary);
    if (!ofs.good())
        throw std::runtime_error("Unable to open positional map file for writing: " + posMapFile);
    
    // Write header: number of rows and columns.
    size_t numRows = posMap.size();
    size_t numCols = (numRows == 0 ? 0 : posMap[0].second.size() + 1);
    
    // Calculate buffer size:
    // header = sizeof(numRows) + sizeof(numCols)
    // for each row: sizeof(std::streampos) + (numCols - 1)*sizeof(uint16_t)
    size_t bufSize = sizeof(numRows) + sizeof(numCols) + numRows * (sizeof(std::streampos) + (numCols - 1) * sizeof(uint16_t));
    std::vector<char> buffer(bufSize);
    
    size_t offset = 0;
    // Copy header data
    std::memcpy(buffer.data() + offset, &numRows, sizeof(numRows));
    offset += sizeof(numRows);
    std::memcpy(buffer.data() + offset, &numCols, sizeof(numCols));
    offset += sizeof(numCols);
    
    // Copy each row's block.
    for (const auto& row : posMap) {
        std::memcpy(buffer.data() + offset, &row.first, sizeof(row.first));
        offset += sizeof(row.first);
        // Write the relative offsets available.
        for (uint16_t rel : row.second) {
            std::memcpy(buffer.data() + offset, &rel, sizeof(uint16_t));
            offset += sizeof(uint16_t);
        }
    }
    
    ofs.write(buffer.data(), bufSize);
    ofs.close();
    std::cout << "posmap write time: " 
              << std::chrono::duration_cast<std::chrono::duration<double>>(clock::now() - time).count() 
              << std::endl;
}

// Updated readPositionalMap: read the entire file into a contiguous buffer then parse.
std::vector<std::pair<std::streampos, std::vector<uint16_t>>>
readPositionalMap(const char* filename) {
    using clock = std::chrono::high_resolution_clock;
    auto time = clock::now();
    
    std::string posMapFile = getPosMapFile(filename);
    std::ifstream ifs(posMapFile, std::ios::binary);
    if (!ifs.good())
        throw std::runtime_error("Cannot open posMap file");
    
    // Get file size.
    ifs.seekg(0, std::ios::end);
    size_t fileSize = static_cast<size_t>(ifs.tellg());
    ifs.seekg(0, std::ios::beg);
    
    std::vector<char> buffer(fileSize);
    ifs.read(buffer.data(), fileSize);
    ifs.close();
    
    size_t offset = 0;
    size_t numRows, numCols;
    std::memcpy(&numRows, buffer.data() + offset, sizeof(numRows));
    offset += sizeof(numRows);
    std::memcpy(&numCols, buffer.data() + offset, sizeof(numCols));
    offset += sizeof(numCols);
    
    std::vector<std::pair<std::streampos, std::vector<uint16_t>>> posMap;
    posMap.resize(numRows);
    
    for (size_t r = 0; r < numRows; r++) {
        std::streampos base;
        std::memcpy(&base, buffer.data() + offset, sizeof(base));
        offset += sizeof(base);
        std::vector<uint16_t> relOffsets(numCols - 1);
        if(numCols > 1) {
            std::memcpy(relOffsets.data(), buffer.data() + offset, (numCols - 1) * sizeof(uint16_t));
            offset += (numCols - 1) * sizeof(uint16_t);
        }
        posMap[r] = std::make_pair(base, relOffsets);
    }

    std::cout << "posmap read time: " 
              << std::chrono::duration_cast<std::chrono::duration<double>>(clock::now() - time).count() 
              << std::endl;
    return posMap;
}