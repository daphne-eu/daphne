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
void writePositionalMap(const char* filename, 
                        const std::vector<std::pair<std::streampos, std::vector<std::uint32_t>>>& posMap) {
    
    std::string posMapFile = getPosMapFile(filename);
    std::ofstream ofs(posMapFile, std::ios::binary);
    if (!ofs.good())
        throw std::runtime_error("Unable to open positional map file for writing: " + posMapFile);
    
    // Write the number of rows.
    size_t numRows = posMap.size();
    ofs.write(reinterpret_cast<const char*>(&numRows), sizeof(numRows));
    
    // For each row, we expect (numCols = relative offsets count + 1) columns.
    size_t numCols = (numRows == 0 ? 0 : posMap[0].second.size() + 1);
    ofs.write(reinterpret_cast<const char*>(&numCols), sizeof(numCols));
    
    // Write for each row:
    // - the absolute offset (base)
    // - follow by (numCols - 1) relative offsets stored as uint32_t.
    for (const auto& row : posMap) {
        ofs.write(reinterpret_cast<const char*>(&row.first), sizeof(row.first));
        for (uint32_t offset : row.second) {
            ofs.write(reinterpret_cast<const char*>(&offset), sizeof(uint32_t));
        }
    }
    ofs.close();
}

// Updated readPositionalMap: reconstruct full offsets.
std::vector<std::pair<std::streampos, std::vector<uint32_t>>>
readPositionalMap(const char* filename) {
    std::ifstream ifs(getPosMapFile(filename), std::ios::binary);
    if (!ifs.good())
        throw std::runtime_error("Cannot open posMap file");
    
    size_t numRows, numCols;
    ifs.read(reinterpret_cast<char*>(&numRows), sizeof(numRows));
    ifs.read(reinterpret_cast<char*>(&numCols), sizeof(numCols));
    
    std::vector<std::pair<std::streampos, std::vector<uint32_t>>> posMap;
    posMap.resize(numRows);
    // For each row, read the base offset and the relative offsets.
    for (size_t r = 0; r < numRows; r++) {
        std::streampos base;
        ifs.read(reinterpret_cast<char*>(&base), sizeof(base));
        std::vector<uint32_t> relOffsets(numCols - 1);
        for (size_t c = 0; c < numCols - 1; c++) {
            uint32_t rel;
            ifs.read(reinterpret_cast<char*>(&rel), sizeof(rel));
            relOffsets[c] = rel;
        }
        posMap[r] = std::make_pair(base, relOffsets);
    }
    return posMap;
}