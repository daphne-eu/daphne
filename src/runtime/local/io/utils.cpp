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
void writePositionalMap(const char *filename, const std::vector<std::vector<std::streampos>> &posMap) {
    std::ofstream posMapFile(std::string(filename) + ".posmap", std::ios::binary);
    if (!posMapFile.is_open()) {
        throw std::runtime_error("Failed to open positional map file");
    }

    for (const auto &colPositions : posMap) {
        for (const auto &pos : colPositions) {
            posMapFile.write(reinterpret_cast<const char *>(&pos), sizeof(pos));
        }
    }

    posMapFile.close();
}

// Function to read or create the positional map
std::vector<std::vector<std::streampos>> readPositionalMap(const char *filename, size_t numCols) {
    std::ifstream posMapFile(std::string(filename) + ".posmap", std::ios::binary);
    if (!posMapFile.is_open()) {
        std::cerr << "Positional map file not found, creating a new one." << std::endl;
        return std::vector<std::vector<std::streampos>>(numCols);
    }
    posMapFile.seekg(0, std::ios::end);
    auto fileSize = posMapFile.tellg();
    posMapFile.seekg(0, std::ios::beg);
    size_t totalEntries = fileSize / sizeof(std::streampos);
    if (totalEntries % numCols != 0) {
        throw std::runtime_error("Incorrect number of entries in posmap file");
    }
    size_t numRows = totalEntries / numCols;
    std::vector<std::vector<std::streampos>> posMap(numCols, std::vector<std::streampos>(numRows));
    // Read in column-major order:
    for (size_t col = 0; col < numCols; col++) {
        for (size_t i = 0; i < numRows; i++) {
            posMap[col][i] = 0;
            posMapFile.read(reinterpret_cast<char *>(&posMap[col][i]), sizeof(std::streampos));
        }
    }
    posMapFile.close();
    return posMap;
}