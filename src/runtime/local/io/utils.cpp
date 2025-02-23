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
// FlatPosMap holds our flattened posmap.
// • numRows and numCols are stored in the header.
// • rowOffsets points to a contiguous block of numRows uint64_t values.
// • relOffsets points to a contiguous block of numRows*numCols uint16_t values.
// The buffer member keeps the allocated memory alive.

// Writes the positional map to a file as two flattened arrays.
// The file layout is as follows:
//   [ header: numRows (uint64_t), numCols (uint64_t) ]
//   [ rowOffsets: numRows * uint64_t ]
//   [ relOffsets: (numRows * numCols +1) * uint16_t ]

void writePositionalMap(const char *filename, size_t numRows, size_t numCols, const uint64_t *rowOffsets,
                        const uint16_t *relOffsets) {
    using clock = std::chrono::high_resolution_clock;
    auto writeTime = clock::now();
    // Layout to write:
    // Header: numRows (uint64_t) followed by numCols (uint64_t)
    // Then: rowOffsets array (numRows * sizeof(uint64_t))
    // Then: relOffsets array ((numRows*numCols + 1) * sizeof(uint16_t))
    size_t headerSize = 2 * sizeof(uint64_t);
    size_t rowArraySize = numRows * sizeof(uint64_t);
    // The flattened relOffsets array must have (numRows * numCols) + 1 entries.
    size_t relArraySize = (numRows * numCols + 1) * sizeof(uint16_t);
    size_t totalSize = headerSize + rowArraySize + relArraySize;

    std::vector<char> buffer(totalSize);
    size_t offset = 0;

    // Write header.
    std::memcpy(buffer.data() + offset, &numRows, sizeof(uint64_t));
    offset += sizeof(uint64_t);
    std::memcpy(buffer.data() + offset, &numCols, sizeof(uint64_t));
    offset += sizeof(uint64_t);

    // Write row offsets.
    std::memcpy(buffer.data() + offset, rowOffsets, rowArraySize);
    offset += rowArraySize;

    // Write flattened relative offsets.
    std::memcpy(buffer.data() + offset, relOffsets, relArraySize);
    // offset += relArraySize;

    std::string posmapFile = getPosMapFile(filename);
    std::ofstream ofs(posmapFile, std::ios::binary);
    if (!ofs)
        throw std::runtime_error("Unable to open posmap file for writing: " + posmapFile);

    ofs.write(buffer.data(), totalSize);
    ofs.flush();
    ofs.close();
    std::cout << "OPERATION=write_posmap,WRITE_TIME="
              << std::chrono::duration_cast<std::chrono::duration<double>>(clock::now() - writeTime).count()
              << std::endl;
    std::cout.flush();
}

PosMap readPositionalMap(const char *filename) {
    using clock = std::chrono::high_resolution_clock;
    auto readTime = clock::now();
    std::string posmapFile = getPosMapFile(filename);
    std::ifstream ifs(posmapFile, std::ios::binary | std::ios::ate);
    if (!ifs)
        throw std::runtime_error("Unable to open posmap file for reading: " + posmapFile);

    std::streamsize size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::vector<char> buffer(static_cast<size_t>(size));
    if (!ifs.read(buffer.data(), size))
        throw std::runtime_error("Failed to read posmap file: " + posmapFile);
    ifs.close();

    size_t offset = 0;
    uint64_t numRows = 0, numCols = 0;
    std::memcpy(&numRows, buffer.data() + offset, sizeof(uint64_t));
    offset += sizeof(uint64_t);
    std::memcpy(&numCols, buffer.data() + offset, sizeof(uint64_t));
    offset += sizeof(uint64_t);

    const uint64_t *rowOffsets = reinterpret_cast<const uint64_t *>(buffer.data() + offset);
    offset += numRows * sizeof(uint64_t);

    // The relOffsets array length is (numRows * numCols) + 1.
    const uint16_t *relOffsets = reinterpret_cast<const uint16_t *>(buffer.data() + offset);

    PosMap posMap;
    posMap.numRows = numRows;
    posMap.numCols = numCols;
    posMap.rowOffsets = rowOffsets;
    posMap.relOffsets = relOffsets;
    // Move the buffer so that its lifetime is tied to posMap.
    posMap.buffer = std::move(buffer);
    std::cout << "OPERATION=read_posmap,READ_TIME="
              << std::chrono::duration_cast<std::chrono::duration<double>>(clock::now() - readTime).count()
              << std::endl;
    std::cout.flush();
    return posMap;
}