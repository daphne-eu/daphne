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

void writeRelativePosMap(const char* filename,
                         const std::vector<uint32_t>& rowStartMap,
                         const std::vector<std::vector<uint16_t>>& relPosMap) {
    std::string posmapFile = getPosMapFile(filename);
    std::ofstream ofs(posmapFile, std::ios::binary);
    if (!ofs)
        throw std::runtime_error("Could not open file for writing positional map.");
    
    // Write the number of rows.
    uint32_t numRows = static_cast<uint32_t>(rowStartMap.size());
    ofs.write(reinterpret_cast<const char*>(&numRows), sizeof(uint32_t));

    // Write the absolute row start positions.
    ofs.write(reinterpret_cast<const char*>(rowStartMap.data()), numRows * sizeof(uint32_t));

    // Write the number of columns.
    uint32_t numCols = static_cast<uint32_t>(relPosMap.size());
    ofs.write(reinterpret_cast<const char*>(&numCols), sizeof(uint32_t));

    // For each column, write its size (should equal numRows) and then its relative offsets.
    for (const auto &colVec : relPosMap) {
        uint32_t colSize = static_cast<uint32_t>(colVec.size());
        ofs.write(reinterpret_cast<const char*>(&colSize), sizeof(uint32_t));
        ofs.write(reinterpret_cast<const char*>(colVec.data()), colSize * sizeof(uint16_t));
    }
    ofs.close();
}

std::pair<std::vector<uint32_t>, std::vector<std::vector<uint16_t>>>
readRelativePosMap(const char* filename, size_t numRows, size_t numCols) {
    std::string posmapFile = getPosMapFile(filename);
    std::ifstream ifs(posmapFile, std::ios::binary);
    if (!ifs)
        throw std::runtime_error("Could not open positional map file for reading.");

    // Read the number of rows.
    uint32_t storedNumRows = 0;
    ifs.read(reinterpret_cast<char*>(&storedNumRows), sizeof(uint32_t));
    if (storedNumRows != numRows)
        throw std::runtime_error("Row count in positional map does not match expected value.");

    // Read the absolute row start positions.
    std::vector<uint32_t> rowStartMap(storedNumRows);
    ifs.read(reinterpret_cast<char*>(rowStartMap.data()), storedNumRows * sizeof(uint32_t));

    // Read the number of columns.
    uint32_t storedNumCols = 0;
    ifs.read(reinterpret_cast<char*>(&storedNumCols), sizeof(uint32_t));
    if (storedNumCols != numCols)
        throw std::runtime_error("Column count in positional map does not match expected value.");

    // Read the relative offsets per column.
    std::vector<std::vector<uint16_t>> relPosMap(numCols);
    for (size_t c = 0; c < numCols; c++) {
        uint32_t colSize = 0;
        ifs.read(reinterpret_cast<char*>(&colSize), sizeof(uint32_t));
        if (colSize != numRows)
            throw std::runtime_error("Relative mapping size for a column does not match expected number of rows.");
        relPosMap[c].resize(colSize);
        ifs.read(reinterpret_cast<char*>(relPosMap[c].data()), colSize * sizeof(uint16_t));
    }
    ifs.close();
    return {rowStartMap, relPosMap};
}

struct PosMapHeader {
    char magic[4];          // e.g. "PMap"
    uint16_t version;       // currently 1
    uint32_t numRows;       // number of rows
    uint32_t numCols;       // number of columns
    uint8_t offsetSize;     // byte-width for relative offsets (e.g., 2)
};

void writeRelativePosMap(const char* filename, const std::vector<std::vector<std::streampos>>& posMap) {
    std::string posmapFile = getPosMapFile(filename);
    std::ofstream ofs(posmapFile, std::ios::binary);
    if (!ofs.good())
        throw std::runtime_error("Unable to open posMap file for writing.");
    
    // Decide which storage size to use for relative offsets:
    // One row always stores an absolute offset for the first column (we store that as uint32_t)
    // and for every subsequent column, we store relative offset to previous delimiter.
    // Assume that for our CSV, relative offsets fit into uint16_t.
    const uint8_t relSize = 2;  // 2 bytes per relative offset

    uint32_t numRows = static_cast<uint32_t>(posMap[0].size());
    uint32_t numCols = static_cast<uint32_t>(posMap.size());
    PosMapHeader header = { {'P', 'M', 'A', 'P'}, 1, numRows, numCols, relSize };
    ofs.write(reinterpret_cast<const char*>(&header), sizeof(header));

    // Write each row
    for (uint32_t r = 0; r < numRows; r++) {
        // Write the absolute offset for the first column as uint32_t.
        uint32_t absOffset = static_cast<uint32_t>(posMap[0][r]);
        ofs.write(reinterpret_cast<const char*>(&absOffset), sizeof(uint32_t));

        // For remaining columns, store relative offsets.
        for (uint32_t c = 1; c < numCols; c++) {
            // Compute the relative offset from the previous delimiter.
            uint32_t relative = static_cast<uint32_t>(posMap[c][r] - posMap[c - 1][r]);
            // Ensure that the relative offset fits into uint16_t.
            if(relative > std::numeric_limits<uint16_t>::max())
                throw std::runtime_error("Relative offset too large to store in 16 bits.");
            uint16_t shortRel = static_cast<uint16_t>(relative);
            ofs.write(reinterpret_cast<const char*>(&shortRel), sizeof(uint16_t));
        }
    }
    ofs.close();
}

std::vector<std::vector<std::streampos>> readRelativePosMap(const char* filename, size_t expectedCols) {
    std::string posmapFile = getPosMapFile(filename);
    std::ifstream ifs(posmapFile, std::ios::binary);
    if (!ifs.good())
        throw std::runtime_error("Failed to open posMap file for reading.");
    
    PosMapHeader header;
    ifs.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (std::string(header.magic, 4) != "PMap")
        throw std::runtime_error("Invalid posMap file format.");
    if (header.numCols != expectedCols)
        throw std::runtime_error("Column count mismatch in posMap file.");

    size_t numRows = header.numRows;
    size_t numCols = header.numCols;
    std::vector<std::vector<std::streampos>> posMap(numCols, std::vector<std::streampos>(numRows, 0));

    // Read absolute offset of first column per row (stored as uint32_t)
    for (size_t r = 0; r < numRows; r++) {
        uint32_t absOffset;
        ifs.read(reinterpret_cast<char*>(&absOffset), sizeof(uint32_t));
        posMap[0][r] = absOffset;
    }
    // For columns 1..(numCols-1), read relative offsets stored as uint16_t and add them cumulatively.
    for (size_t c = 1; c < numCols; c++) {
        for (size_t r = 0; r < numRows; r++) {
            uint16_t relOffset;
            ifs.read(reinterpret_cast<char*>(&relOffset), sizeof(uint16_t));
            posMap[c][r] = posMap[c-1][r] + static_cast<std::streamoff>(relOffset);
        }
    }
    return posMap;
}
