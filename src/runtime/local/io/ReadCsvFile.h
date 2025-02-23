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
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>

#include <runtime/local/io/utils.h>

#include <util/preprocessor_defs.h>

#include <type_traits>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <queue>
#include <sstream>
#include <stdexcept>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes> struct ReadCsvFile {
    static void apply(DTRes *&res, File *file, size_t numRows, size_t numCols, char delim,
                      const char *filename = nullptr, bool usePosMap = false) = delete;

    static void apply(DTRes *&res, File *file, size_t numRows, size_t numCols, ssize_t numNonZeros, bool sorted = true,
                      const char *filename = nullptr, bool usePosMap = false) = delete;

    static void apply(DTRes *&res, File *file, size_t numRows, size_t numCols, char delim, ValueTypeCode *schema,
                      const char *filename = nullptr, bool usePosMap = false) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes>
void readCsvFile(DTRes *&res, File *file, size_t numRows, size_t numCols, char delim, const char *filename = nullptr,
                 bool usePosMap = false) {
    ReadCsvFile<DTRes>::apply(res, file, numRows, numCols, delim, filename, usePosMap);
}

template <class DTRes>
void readCsvFile(DTRes *&res, File *file, size_t numRows, size_t numCols, char delim, ValueTypeCode *schema,
                 const char *filename = nullptr, bool usePosMap = false) {
    ReadCsvFile<DTRes>::apply(res, file, numRows, numCols, delim, schema, filename, usePosMap);
}

template <class DTRes>
void readCsvFile(DTRes *&res, File *file, size_t numRows, size_t numCols, char delim, ssize_t numNonZeros,
                 bool sorted = true, const char *filename = nullptr, bool usePosMap = false) {
    ReadCsvFile<DTRes>::apply(res, file, numRows, numCols, delim, numNonZeros, sorted, filename, usePosMap);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct ReadCsvFile<DenseMatrix<VT>> {
    static void apply(DenseMatrix<VT> *&res, struct File *file, size_t numRows, size_t numCols, char delim,
                      const char *filename = nullptr, bool usePosMap = false) {
        if (file == nullptr)
            throw std::runtime_error("ReadCsvFile: requires a file to be specified (must not be nullptr)");
        if (numRows <= 0)
            throw std::runtime_error("ReadCsvFile: numRows must be > 0");
        if (numCols <= 0)
            throw std::runtime_error("ReadCsvFile: numCols must be > 0");

        if (res == nullptr) {
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false);
        }

        size_t cell = 0;
        VT *valuesRes = res->getValues();
        bool posMapExists = false;
        PosMap posMap;
        using clock = std::chrono::high_resolution_clock;
        auto time = clock::now();
        // Optimized branch using positional map.
        if (usePosMap) {
            // Read the positional map from file.
            try {
                posMap = readPositionalMap(filename);
                usePosMap = true;
            } catch (std::exception &e) {
                // try to create posMap
            }
        }
        if (posMapExists) {
            std::ifstream ifs(filename, std::ios::binary);
            if (!ifs.good())
                throw std::runtime_error("Optimized branch: failed to open file for in-memory buffering");
            std::vector<char> fileBuffer((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
            // Build row pointers using absolute row offsets from the posmap.
            std::vector<const char *> rowPointers(numRows);
            for (size_t r = 0; r < numRows; r++) {
                rowPointers[r] = fileBuffer.data() + static_cast<size_t>(posMap.rowOffsets[r]);
            }
            // For each row, use stored relative offsets to extract each field.
            for (size_t r = 0; r < numRows; r++) {
                auto baseOffset = posMap.rowOffsets[r];
                const char *linePtr = rowPointers[r];
                const uint16_t *relOffsets = posMap.relOffsets + (r * numCols);
                std::vector<size_t> nextPosArr(numCols);
                for (size_t c = 0; c < numCols; c++) {
                    if (c < numCols - 1)
                        nextPosArr[c] = static_cast<size_t>(relOffsets[c + 1]);
                    else if (r < numRows - 1)
                        nextPosArr[c] = static_cast<size_t>(posMap.rowOffsets[r + 1]) - baseOffset;
                    else
                        nextPosArr[c] = fileBuffer.size() - baseOffset;
                }
                for (size_t c = 0; c < numCols; c++) {
                    size_t pos = relOffsets[c];
                    size_t nextPos = nextPosArr[c];
                    std::string field(linePtr + pos, nextPos - pos - 1);
                    VT val;
                    convertCstr(field.c_str(), &val);
                    valuesRes[cell++] = val;
                }
            }

            std::cout << "READ_TYPE=second,READ_TIME="
                      << std::chrono::duration_cast<std::chrono::duration<double>>(clock::now() - time).count()
                      << std::endl;
            std::cout.flush();
            return;
        }

        if (usePosMap) {
            auto *rowOffsets = new uint64_t[numRows];
            auto *relOffsets = new uint16_t[numRows * numCols + 1];
            uint64_t currentPos = 0;
            for (size_t r = 0; r < numRows; r++) {
                ssize_t ret = getFileLine(file);
                if ((file->read == EOF) || (file->line == NULL))
                    break;
                if (ret == -1)
                    throw std::runtime_error("ReadCsvFile::apply: getFileLine failed");
                // Record the absolute offset for this row.
                rowOffsets[r] = currentPos;
                relOffsets[r * numCols] = 0;
                size_t pos = 0;
                for (size_t c = 0; c < numCols; c++) {
                    VT val;
                    convertCstr(file->line + pos, &val);
                    valuesRes[cell++] = val;
                    if (c < numCols - 1) {
                        // Advance pos until the delimiter is found.
                        while (file->line[pos] != delim)
                            pos++;
                        pos++; // skip delimiter
                        relOffsets[r * numCols + c + 1] = static_cast<uint16_t>(pos);
                    }
                }
                currentPos = static_cast<uint64_t>(file->pos);
            }
            relOffsets[numRows * numCols] =
                static_cast<uint16_t>(currentPos - rowOffsets[numRows - 1]); // end of last field
            std::cout << "READ_TYPE=first,READ_TIME="
                      << std::chrono::duration_cast<std::chrono::duration<double>>(clock::now() - time).count()
                      << std::endl;
            std::cout.flush();
            try {
                writePositionalMap(filename, numRows, numCols, rowOffsets, relOffsets);
            } catch (std::exception &e) {
                // Even if posmap writing fails, parsing was successful.
            }
            delete[] rowOffsets;
            delete[] relOffsets;
        } else {
            for (size_t r = 0; r < numRows; r++) {
                if (getFileLine(file) == -1)
                    throw std::runtime_error("ReadCsvFile::apply: getFileLine failed");
                size_t pos = 0;
                for (size_t c = 0; c < numCols; c++) {
                    VT val;
                    convertCstr(file->line + pos, &val);
                    valuesRes[cell++] = val;
                    if (c < numCols - 1) {
                        while (file->line[pos] != delim)
                            pos++;
                        pos++; // skip delimiter
                    }
                }
            }
            std::cout << "READ_TYPE=normal,READ_TIME="
                      << std::chrono::duration_cast<std::chrono::duration<double>>(clock::now() - time).count()
                      << std::endl;
            std::cout.flush();
        }
    }
};

template <> struct ReadCsvFile<DenseMatrix<std::string>> {
    static void apply(DenseMatrix<std::string> *&res, struct File *file, size_t numRows, size_t numCols, char delim,
                      const char *filename = nullptr, bool usePosMap = false) {
        if (file == nullptr)
            throw std::runtime_error("ReadCsvFile: requires a file to be specified (must not be nullptr)");
        if (numRows <= 0)
            throw std::runtime_error("ReadCsvFile: numRows must be > 0");
        if (numCols <= 0)
            throw std::runtime_error("ReadCsvFile: numCols must be > 0");

        if (res == nullptr) {
            res = DataObjectFactory::create<DenseMatrix<std::string>>(numRows, numCols, false);
        }

        // non-optimized branch (unchanged)
        size_t cell = 0;
        std::string *valuesRes = res->getValues();
        using clock = std::chrono::high_resolution_clock;
        auto time = clock::now();
        bool posMapExists = false;
        PosMap posMap;
        // Optimized branch using positional map.
        if (usePosMap) {
            // Read the positional map from file.
            try {
                posMap = readPositionalMap(filename);
                posMapExists = true;
            } catch (std::exception &e) {
                // try to create posMap
            }
        }
        if (posMapExists) {
            auto t0 = clock::now();
            std::ifstream ifs(filename, std::ios::binary);
            if (!ifs.good())
                throw std::runtime_error("Optimized branch: failed to open file for in-memory buffering");
            std::vector<char> fileBuffer((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
            // Build row pointers from posMap offsets.
            // auto t1 = clock::now();
            // std::cout << "Time to load file into buffer: "
            //     << std::chrono::duration_cast<std::chrono::duration<double>>(t1-t0).count() << " s" << std::endl;
            //     std::cout.flush();

            std::vector<const char *> rowPointers(numRows);
            for (size_t r = 0; r < numRows; r++) {
                rowPointers[r] = fileBuffer.data() + static_cast<size_t>(posMap.rowOffsets[r]);
            }
            // auto t2 = clock::now();
            // std::cout << "Time to build row pointers: "
            //  << std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1).count() << " s" << std::endl;
            //  std::cout.flush();

            // For each row, use the relative offsets stored in posMap.
            // For each row, precompute the nextPos for each field.
            for (size_t r = 0; r < numRows; r++) {
                auto baseOffset = posMap.rowOffsets[r];
                const char *linePtr = rowPointers[r];
                // Compute pointer for relative offsets for row r.
                const uint16_t *relOffsets = posMap.relOffsets + (r * numCols);
                // Precompute boundaries for every field in this row.
                std::vector<size_t> nextPosArr(numCols);
                for (size_t c = 0; c < numCols; c++) {
                    if (c < numCols - 1)
                        nextPosArr[c] = static_cast<size_t>(relOffsets[c + 1]);
                    else if (r < numRows - 1)
                        nextPosArr[c] = static_cast<size_t>(posMap.rowOffsets[r + 1]) - baseOffset;
                    else
                        nextPosArr[c] = fileBuffer.size() - baseOffset; // for the last row
                }
                // Extract all fields using the precomputed boundaries.
                for (size_t c = 0; c < numCols; c++) {
                    size_t pos = relOffsets[c]; // relative to linePtr
                    size_t nextPos = nextPosArr[c];
                    std::string val;
                    // Pass start_pos as 0 since linePtr+pos already points to the field start.
                    setCString(linePtr + pos, &val, delim, nextPos - pos - 1);
                    valuesRes[cell++] = val;
                }
            }
            auto t3 = clock::now();
            // std::cout << "Time for field extraction (posmap branch): "
            //<< std::chrono::duration_cast<std::chrono::duration<double>>(t3-t2).count() << " s" << std::endl;
            //std::cout.flush();
            std::cout << "READ_TYPE=second,READ_TIME="
                      << std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t0).count() << " s"
                      << std::endl;
            std::cout.flush();
            return;
        }
        if (usePosMap) {
            auto *rowOffsets = new uint64_t[numRows];
            auto *relOffsets = new uint16_t[numRows * numCols + 1];
            uint64_t currentPos = 0;
            for (size_t r = 0; r < numRows; r++) {
                ssize_t ret = getFileLine(file);
                if ((file->read == EOF) || (file->line == NULL))
                    break;
                if (ret == -1)
                    throw std::runtime_error("ReadCsvFile::apply: getFileLine failed");
                // Record the absolute offset for this row.
                rowOffsets[r] = currentPos;
                relOffsets[r * numCols] = 0;
                size_t offset = 0;
                size_t pos = 0;
                for (size_t c = 0; c < numCols; c++) {
                    std::string val("");
                    // Here we call the file–based setCString (which advances pos and updates offset)
                    pos = setCString(file, pos, &val, delim, &offset);
                    valuesRes[cell++] = val;
                    if (c < numCols - 1) {
                        // Advance pos until we hit the delimiter.
                        while (file->line[pos] != delim)
                            pos++;
                        pos++; // skip delimiter
                    }
                    // Record relative offsets (including multi-line offset adjustments)
                    if (c < numCols - 1) {
                        if (offset > 0)
                            relOffsets[r * numCols + c + 1] = static_cast<uint16_t>(pos + offset);
                        else
                            relOffsets[r * numCols + c + 1] = static_cast<uint16_t>(pos);
                    }
                }
                currentPos = static_cast<uint64_t>(file->pos);
            }
            relOffsets[numRows * numCols] =
                static_cast<uint16_t>(currentPos - rowOffsets[numRows - 1]); // end of last field
            try {
                writePositionalMap(filename, numRows, numCols, rowOffsets, relOffsets);
            } catch (std::exception &e) {
                // If writing fails, posmap may still be used later.
            }
            delete[] rowOffsets;
            delete[] relOffsets;
            std::cout << "READ_TYPE=first,READ_TIME="
                      << std::chrono::duration_cast<std::chrono::duration<double>>(clock::now() - time).count()
                      << std::endl;
            std::cout.flush();
            return;
        } else {
            for (size_t r = 0; r < numRows; r++) {
                if (getFileLine(file) == -1)
                    throw std::runtime_error("ReadCsvFile::apply: getFileLine failed");

                size_t pos = 0;
                size_t offset = 0;
                for (size_t c = 0; c < numCols; c++) {
                    std::string val("");
                    pos = setCString(file, pos, &val, delim, &offset) + 1;
                    valuesRes[cell++] = val;
                }
            }
            std::cout << "READ_TYPE=normal,READ_TIME="
                      << std::chrono::duration_cast<std::chrono::duration<double>>(clock::now() - time).count()
                      << std::endl;
            std::cout.flush();
        }
    }
};

template <> struct ReadCsvFile<DenseMatrix<FixedStr16>> {
    static void apply(DenseMatrix<FixedStr16> *&res, struct File *file, size_t numRows, size_t numCols, char delim,
                      const char *filename = nullptr, bool usePosMap = false) {
        if (file == nullptr)
            throw std::runtime_error("ReadCsvFile: requires a file to be specified (must not be nullptr)");
        if (numRows <= 0)
            throw std::runtime_error("ReadCsvFile: numRows must be > 0");
        if (numCols <= 0)
            throw std::runtime_error("ReadCsvFile: numCols must be > 0");

        if (res == nullptr) {
            res = DataObjectFactory::create<DenseMatrix<FixedStr16>>(numRows, numCols, false);
        }
        using clock = std::chrono::high_resolution_clock;
        auto time = clock::now();
        size_t cell = 0;
        FixedStr16 *valuesRes = res->getValues();
        if (usePosMap) {
            // posMap is stored as: posMap[c][r] = absolute offset for column c, row r.
            // std::vector<std::pair<std::streampos, std::vector<std::uint16_t>>> posMap = readPositionalMap(filename);
            std::ifstream ifs(filename, std::ios::binary);
            if (!ifs.good())
                throw std::runtime_error("Optimized branch: failed to open file for in-memory buffering");
            std::vector<char> fileBuffer((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
            const char *linePtr = fileBuffer.data();
            size_t pos = 0;
            for (size_t r = 0; r < numRows; r++) {
                // For every column, compute the relative offset within the line
                for (size_t c = 0; c < numCols; c++) {
                    size_t nextPos = pos + 16;
                    std::string val;

                    setCString(linePtr + pos, &val, delim, nextPos - pos - 1);
                    valuesRes[cell++].set(val.c_str());
                    pos = nextPos + 1;
                }
            }
            std::cout << "read time optimized: "
                      << std::chrono::duration_cast<std::chrono::duration<double>>(clock::now() - time).count()
                      << std::endl;
            std::cout.flush();
            return;
        }
        for (size_t r = 0; r < numRows; r++) {
            if (getFileLine(file) == -1)
                throw std::runtime_error("ReadCsvFile::apply: getFileLine failed");

            size_t pos = 0;
            size_t offset = 0;
            for (size_t c = 0; c < numCols; c++) {
                std::string val("");
                pos = setCString(file, pos, &val, delim, &offset) + 1;
                valuesRes[cell++].set(val.c_str());
            }
        }
        std::cout << "read time: "
                  << std::chrono::duration_cast<std::chrono::duration<double>>(clock::now() - time).count()
                  << std::endl;
        std::cout.flush();
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct ReadCsvFile<CSRMatrix<VT>> {
    static void apply(CSRMatrix<VT> *&res, struct File *file, size_t numRows, size_t numCols, char delim,
                      ssize_t numNonZeros, bool sorted = true, const char *filename = nullptr, bool usePosMap = false) {
        if (numNonZeros == -1)
            throw std::runtime_error(
                "ReadCsvFile: Currently, reading of sparse matrices requires a number of non zeros to be defined");

        if (res == nullptr)
            res = DataObjectFactory::create<CSRMatrix<VT>>(numRows, numCols, numNonZeros, false);

        if (sorted) {
            readCOOSorted(res, file, numRows, numCols, static_cast<size_t>(numNonZeros), delim);
        } else {
            DenseMatrix<uint64_t> *rowColumnPairs = nullptr;
            readCsvFile(rowColumnPairs, file, static_cast<size_t>(numNonZeros), 2, delim, filename);
            readCOOUnsorted(res, rowColumnPairs, numRows, numCols, static_cast<size_t>(numNonZeros));
            DataObjectFactory::destroy(rowColumnPairs);
        }
    }

  private:
    static void readCOOSorted(CSRMatrix<VT> *&res, File *file, size_t numRows, [[maybe_unused]] size_t numCols,
                              size_t numNonZeros, char delim) {
        auto *rowOffsets = res->getRowOffsets();
        // we first write number of non zeros for each row and then compute the
        // cumulative sum
        std::memset(rowOffsets, 0, (numRows + 1) * sizeof(size_t));
        auto *colIdxs = res->getColIdxs();
        auto *values = res->getValues();

        size_t pos;
        uint64_t row;
        uint64_t col;
        for (size_t i = 0; i < numNonZeros; ++i) {
            if (getFileLine(file) == -1)
                throw std::runtime_error("ReadCOOSorted::apply: getFileLine failed");
            convertCstr(file->line, &row);
            pos = 0;
            while (file->line[pos] != delim)
                pos++;
            pos++; // skip delimiter
            convertCstr(file->line + pos, &col);

            rowOffsets[row + 1] += 1;
            values[i] = 1;
            colIdxs[i] = col;
        }
        //        #pragma clang loop vectorize(enable)
        PRAGMA_LOOP_VECTORIZE
        for (size_t r = 1; r <= numRows; ++r) {
            rowOffsets[r] += rowOffsets[r - 1];
        }
    }

    static void readCOOUnsorted(CSRMatrix<VT> *&res, DenseMatrix<uint64_t> *rowColumnPairs, size_t numRows,
                                size_t numCols, size_t numNonZeros) {
        // pairs are ordered by first then by second argument (row, then col)
        using RowColPos = std::pair<size_t, size_t>;
        std::priority_queue<RowColPos, std::vector<RowColPos>, std::greater<>> positions;
        for (auto r = 0u; r < rowColumnPairs->getNumRows(); ++r) {
            positions.emplace(rowColumnPairs->get(r, 0), rowColumnPairs->get(r, 1));
        }

        auto *rowOffsets = res->getRowOffsets();
        rowOffsets[0] = 0;
        auto *colIdxs = res->getColIdxs();
        auto *values = res->getValues();
        size_t currValIdx = 0;
        size_t rowIdx = 0;
        while (!positions.empty()) {
            auto pos = positions.top();
            if (pos.first >= res->getNumRows() || pos.second >= res->getNumCols()) {
                throw std::runtime_error("Position [" + std::to_string(pos.first) + ", " + std::to_string(pos.second) +
                                         "] is not part of matrix<" + std::to_string(res->getNumRows()) + ", " +
                                         std::to_string(res->getNumCols()) + ">");
            }
            while (rowIdx < pos.first) {
                rowOffsets[rowIdx + 1] = currValIdx;
                rowIdx++;
            }
            // TODO: valued COO files?
            values[currValIdx] = 1;
            colIdxs[currValIdx] = pos.second;
            currValIdx++;
            positions.pop();
        }
        while (rowIdx < numRows) {
            rowOffsets[rowIdx + 1] = currValIdx;
            rowIdx++;
        }
    }
};

// ----------------------------------------------------------------------------
// Frame
// ----------------------------------------------------------------------------

template <> struct ReadCsvFile<Frame> {
    static void apply(Frame *&res, struct File *file, size_t numRows, size_t numCols, char delim, ValueTypeCode *schema,
                      const char *filename, bool usePosMap = false) {
        if (file == nullptr)
            throw std::runtime_error("ReadCsvFile: requires a file to be specified (must not be nullptr): " +
                                     std::string(filename));
        if (numRows <= 0)
            throw std::runtime_error("ReadCsvFile: numRows must be > 0");
        if (numCols <= 0)
            throw std::runtime_error("ReadCsvFile: numCols must be > 0");

        if (res == nullptr) {
            res = DataObjectFactory::create<Frame>(numRows, numCols, schema, nullptr, false);
        }

        uint8_t **rawCols = new uint8_t *[numCols];
        ValueTypeCode *colTypes = new ValueTypeCode[numCols];
        for (size_t i = 0; i < numCols; i++) {
            rawCols[i] = reinterpret_cast<uint8_t *>(res->getColumnRaw(i));
            colTypes[i] = res->getColumnType(i);
        }
        // Determine if any optimized branch should be used.
        bool posMapExists = false;
        std::string fName;
        if (usePosMap && filename) {
            fName = filename;
            std::string posmapFile = getPosMapFile(fName.c_str());
            if (usePosMap && std::filesystem::exists(posmapFile)) {
                posMapExists = true;
                fName = posmapFile;
            }
        }
        using clock = std::chrono::high_resolution_clock;
        auto time = clock::now();

        if (posMapExists) {
            // posMap is stored as: posMap[c][r] = absolute offset for column c, row r.
            PosMap posMap = readPositionalMap(filename);
            std::ifstream ifs(filename, std::ios::binary);
            if (!ifs.good())
                throw std::runtime_error("Optimized branch: failed to open file for in-memory buffering");
            std::vector<char> fileBuffer((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
            std::vector<const char *> rowPointers;
            rowPointers.resize(numRows);
            for (size_t r = 0; r < numRows; r++) {
                // Compute pointer for row r from posMap’s absolute offset.
                rowPointers[r] = fileBuffer.data() + static_cast<size_t>(posMap.rowOffsets[r]);
            }

            for (size_t r = 0; r < numRows; r++) {
                // Read the entire row by seeking to the beginning of row r (first field)
                auto baseOffset = posMap.rowOffsets[r];
                const char *linePtr = rowPointers[r];
                const uint16_t *relOffsets = posMap.relOffsets + (r * numCols);
                std::vector<size_t> nextPosArr(numCols);
                for (size_t c = 0; c < numCols; c++) {
                    if (c < numCols - 1)
                        nextPosArr[c] = static_cast<size_t>(relOffsets[c + 1]);
                    else if (r < numRows - 1)
                        nextPosArr[c] = static_cast<size_t>(posMap.rowOffsets[r + 1]) - baseOffset;
                    else
                        nextPosArr[c] = fileBuffer.size() - baseOffset;
                }
                // For every column, compute the relative offset within the line
                for (size_t c = 0; c < numCols; c++) {
                    size_t pos = relOffsets[c];
                    switch (colTypes[c]) {
                    case ValueTypeCode::SI8: {
                        int8_t val;
                        convertCstr(linePtr + pos, &val);
                        reinterpret_cast<int8_t *>(rawCols[c])[r] = val;
                        break;
                    }
                    case ValueTypeCode::SI32: {
                        int32_t val;
                        convertCstr(linePtr + pos, &val);
                        reinterpret_cast<int32_t *>(rawCols[c])[r] = val;
                        break;
                    }
                    case ValueTypeCode::SI64: {
                        int64_t val;
                        convertCstr(linePtr + pos, &val);
                        reinterpret_cast<int64_t *>(rawCols[c])[r] = val;
                        break;
                    }
                    case ValueTypeCode::UI8: {
                        uint8_t val;
                        convertCstr(linePtr + pos, &val);
                        reinterpret_cast<uint8_t *>(rawCols[c])[r] = val;
                        break;
                    }
                    case ValueTypeCode::UI32: {
                        uint32_t val;
                        convertCstr(linePtr + pos, &val);
                        reinterpret_cast<uint32_t *>(rawCols[c])[r] = val;
                        break;
                    }
                    case ValueTypeCode::UI64: {
                        uint64_t val;
                        convertCstr(linePtr + pos, &val);
                        reinterpret_cast<uint64_t *>(rawCols[c])[r] = val;
                        break;
                    }
                    case ValueTypeCode::F32: {
                        float val;
                        convertCstr(linePtr + pos, &val);
                        reinterpret_cast<float *>(rawCols[c])[r] = val;
                        break;
                    }
                    case ValueTypeCode::F64: {
                        double val;
                        convertCstr(linePtr + pos, &val);
                        reinterpret_cast<double *>(rawCols[c])[r] = val;
                        break;
                    }
                    case ValueTypeCode::STR: {
                        std::string val;
                        setCString(linePtr + pos, &val, delim,
                                   nextPosArr[c] - pos - 1); // needed for double quote encoding
                        reinterpret_cast<std::string *>(rawCols[c])[r] = val;
                        break;
                    }
                    case ValueTypeCode::FIXEDSTR16: {
                        std::string val;
                        setCString(linePtr + pos, &val, delim,
                                   nextPosArr[c] - pos - 1); // not passing delimiter to nextPos
                        reinterpret_cast<std::string *>(rawCols[c])[r] = val;
                        break;
                    }
                    default:
                        throw std::runtime_error("ReadCsvFile::apply: unknown value type code");
                    }
                }
            }
            delete[] rawCols;
            delete[] colTypes;
            std::cout << "READ_TYPE=second,READ_TIME="
                      << std::chrono::duration_cast<std::chrono::duration<double>>(clock::now() - time).count()
                      << std::endl;
            std::cout.flush();
            return;
        }

        // Normal branch: iterate row by row and for each field save its absolute offset.
        auto *rowOffsets = new uint64_t[numRows];
        auto *relOffsets = new uint16_t[numRows * numCols + 1];

        uint64_t currentPos = 0;
        for (size_t row = 0; row < numRows; row++) {
            ssize_t ret = getFileLine(file);
            if ((file->read == EOF) || (file->line == NULL))
                break;
            if (ret == -1)
                throw std::runtime_error("ReadCsvFile::apply: getFileLine failed");

            // Save absolute offset for this row.
            if (usePosMap) {
                rowOffsets[row] = currentPos;
                relOffsets[row * numCols] = static_cast<uint16_t>(0);
            }
            size_t offset = 0;
            size_t pos = 0;
            for (size_t col = 0; col < numCols; col++) {
                switch (colTypes[col]) {
                case ValueTypeCode::SI8:
                    int8_t val_si8;
                    convertCstr(file->line + pos, &val_si8);
                    reinterpret_cast<int8_t *>(rawCols[col])[row] = val_si8;
                    break;
                case ValueTypeCode::SI32:
                    int32_t val_si32;
                    convertCstr(file->line + pos, &val_si32);
                    reinterpret_cast<int32_t *>(rawCols[col])[row] = val_si32;
                    break;
                case ValueTypeCode::SI64:
                    int64_t val_si64;
                    convertCstr(file->line + pos, &val_si64);
                    reinterpret_cast<int64_t *>(rawCols[col])[row] = val_si64;
                    break;
                case ValueTypeCode::UI8:
                    uint8_t val_ui8;
                    convertCstr(file->line + pos, &val_ui8);
                    reinterpret_cast<uint8_t *>(rawCols[col])[row] = val_ui8;
                    break;
                case ValueTypeCode::UI32:
                    uint32_t val_ui32;
                    convertCstr(file->line + pos, &val_ui32);
                    reinterpret_cast<uint32_t *>(rawCols[col])[row] = val_ui32;
                    break;
                case ValueTypeCode::UI64:
                    uint64_t val_ui64;
                    convertCstr(file->line + pos, &val_ui64);
                    reinterpret_cast<uint64_t *>(rawCols[col])[row] = val_ui64;
                    break;
                case ValueTypeCode::F32:
                    float val_f32;
                    convertCstr(file->line + pos, &val_f32);
                    reinterpret_cast<float *>(rawCols[col])[row] = val_f32;
                    break;
                case ValueTypeCode::F64:
                    double val_f64;
                    convertCstr(file->line + pos, &val_f64);
                    reinterpret_cast<double *>(rawCols[col])[row] = val_f64;
                    break;
                case ValueTypeCode::STR: {
                    std::string val_str = "";
                    pos = setCString(file, pos, &val_str, delim, &offset);
                    reinterpret_cast<std::string *>(rawCols[col])[row] = val_str;
                    break;
                }
                case ValueTypeCode::FIXEDSTR16: {
                    std::string val_str = "";
                    pos = setCString(file, pos, &val_str, delim, &offset);
                    reinterpret_cast<FixedStr16 *>(rawCols[col])[row] = FixedStr16(val_str);
                    break;
                }
                default:
                    throw std::runtime_error("ReadCsvFile::apply: unknown value type code");
                }

                if (col < numCols - 1) {
                    // Advance pos until next delimiter
                    while (file->line[pos] != delim)
                        pos++;
                    pos++; // skip delimiter
                }

                if (usePosMap) {
                    if (col < numCols - 1) {
                        if (offset > 0) {
                            relOffsets[row * numCols + col + 1] =
                                static_cast<uint16_t>(pos + offset); // adds offset from possible multiline string
                        } else
                            relOffsets[row * numCols + col + 1] = static_cast<uint16_t>(pos);
                    }
                }
            }
            currentPos = static_cast<uint64_t>(file->pos);
        }
        relOffsets[numRows * numCols] =
            static_cast<uint16_t>(currentPos - rowOffsets[numRows - 1]); // end of last element
        std::string message = (usePosMap) ? "READ_TYPE=first,READ_TIME=" : "READ_TYPE=normal,READ_TIME=";
        std::cout << message << std::chrono::duration_cast<std::chrono::duration<double>>(clock::now() - time).count()
                  << std::endl;
        std::cout.flush();

        if (usePosMap) {
            try {
                writePositionalMap(filename, numRows, numCols, rowOffsets, relOffsets);
            } catch (std::exception &e) {
                // positional map can still be used
            }
        }
        delete[] rawCols;
        delete[] colTypes;
    }
};