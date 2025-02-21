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

struct ReadOpts {
    bool opt_enabled;
    bool posMap;
    bool useDoubleQuoteEncode;

    explicit ReadOpts(bool opt_enabled = false, bool posMap = true, bool useDoubleQuoteEncode = true) : opt_enabled(opt_enabled), posMap(posMap), useDoubleQuoteEncode(useDoubleQuoteEncode) {}
};

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes> struct ReadCsvFile {
    static void apply(DTRes *&res, File *file, size_t numRows, size_t numCols, char delim,
                      const char *filename = nullptr, ReadOpts opt = ReadOpts()) = delete;

    static void apply(DTRes *&res, File *file, size_t numRows, size_t numCols, ssize_t numNonZeros, bool sorted = true,
                      const char *filename = nullptr, ReadOpts opt = ReadOpts()) = delete;

    static void apply(DTRes *&res, File *file, size_t numRows, size_t numCols, char delim, ValueTypeCode *schema,
                      const char *filename = nullptr, ReadOpts opt = ReadOpts()) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes>
void readCsvFile(DTRes *&res, File *file, size_t numRows, size_t numCols, char delim, const char *filename = nullptr,
                 ReadOpts opt = ReadOpts()) {
    ReadCsvFile<DTRes>::apply(res, file, numRows, numCols, delim, filename, opt);
}

template <class DTRes>
void readCsvFile(DTRes *&res, File *file, size_t numRows, size_t numCols, char delim, ValueTypeCode *schema,
                 const char *filename = nullptr, ReadOpts opt = ReadOpts()) {
    ReadCsvFile<DTRes>::apply(res, file, numRows, numCols, delim, schema, filename, opt);
}

template <class DTRes>
void readCsvFile(DTRes *&res, File *file, size_t numRows, size_t numCols, char delim, ssize_t numNonZeros,
                 bool sorted = true, const char *filename = nullptr, ReadOpts opt = ReadOpts()) {
    ReadCsvFile<DTRes>::apply(res, file, numRows, numCols, delim, numNonZeros, sorted, filename, opt);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct ReadCsvFile<DenseMatrix<VT>> {
    static void apply(DenseMatrix<VT> *&res, struct File *file, size_t numRows, size_t numCols, char delim,
                      const char *filename = nullptr, ReadOpts opt = ReadOpts()) {
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
    }
};

template <> struct ReadCsvFile<DenseMatrix<std::string>> {
    static void apply(DenseMatrix<std::string> *&res, struct File *file, size_t numRows, size_t numCols, char delim,
                      const char *filename = nullptr, ReadOpts opt = ReadOpts()) {
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
    }
};

template <> struct ReadCsvFile<DenseMatrix<FixedStr16>> {
    static void apply(DenseMatrix<FixedStr16> *&res, struct File *file, size_t numRows, size_t numCols, char delim,
                      const char *filename = nullptr, ReadOpts opt = ReadOpts()) {
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
        if (opt.opt_enabled && opt.posMap) {
            // posMap is stored as: posMap[c][r] = absolute offset for column c, row r.
            //std::vector<std::pair<std::streampos, std::vector<std::uint16_t>>> posMap = readPositionalMap(filename);
            std::ifstream ifs(filename, std::ios::binary);
            if (!ifs.good())
                throw std::runtime_error("Optimized branch: failed to open file for in-memory buffering");
            std::vector<char> fileBuffer((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
            const char* linePtr = fileBuffer.data();
            size_t pos = 0;
            for (size_t r = 0; r < numRows; r++) {
                // For every column, compute the relative offset within the line
                for (size_t c = 0; c < numCols; c++) {
                    size_t nextPos = pos + 16;

                    const char posChar = (linePtr + pos)[0];
                    const char nextPosChar = (linePtr + nextPos - 2)[0];
                    if ((nextPos - pos > 0) && posChar == '\"' && nextPosChar == '\"') { // remove quotes
                        pos += 1;
                        nextPos -= 1;
                    }
                    std::string val(linePtr + pos, nextPos - pos - 1);
                    if (opt.useDoubleQuoteEncode) {
                        valuesRes[cell++].set(convertDoubleQuotes(val).c_str());
                    } else
                        valuesRes[cell++].set(val.c_str());
                    pos = nextPos + 1;
                }
            }
            std::cout << "read time optimized2: " << std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - time).count() << " ms" << std::endl;
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
        std::cout << "read time2: " << std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - time).count() << " ms" << std::endl;
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct ReadCsvFile<CSRMatrix<VT>> {
    static void apply(CSRMatrix<VT> *&res, struct File *file, size_t numRows, size_t numCols, char delim,
                      ssize_t numNonZeros, bool sorted = true, const char *filename = nullptr,
                      ReadOpts opt = ReadOpts()) {
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
                      const char *filename, ReadOpts opt = ReadOpts()) {
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
        bool useOptimized = false;
        bool usePosMap = false;
        std::string fName;
        if (opt.opt_enabled && filename) {
            fName = filename;
            std::string posmapFile = getPosMapFile(fName.c_str());
            if (opt.posMap && std::filesystem::exists(posmapFile)) {
                useOptimized = true;
                usePosMap = true;
                fName = posmapFile;
            }
        }
        // using clock = std::chrono::high_resolution_clock;
        // auto time = clock::now();
        if (useOptimized) {
            if (usePosMap) {
                // posMap is stored as: posMap[c][r] = absolute offset for column c, row r.
                std::vector<std::pair<std::streampos, std::vector<std::uint16_t>>> posMap = readPositionalMap(filename);
                std::ifstream ifs(filename, std::ios::binary);
                if (!ifs.good())
                    throw std::runtime_error("Optimized branch: failed to open file for in-memory buffering");
                std::vector<char> fileBuffer((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
                for (size_t r = 0; r < numRows; r++) {
                    // Read the entire row by seeking to the beginning of row r (first field)
                    size_t baseOffset = static_cast<size_t>(posMap[r].first);
                    const char *linePtr = fileBuffer.data() + baseOffset;

                    // For every column, compute the relative offset within the line
                    for (size_t c = 0; c < numCols; c++) {
                        size_t pos = static_cast<size_t>(posMap[r].second[c]);
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
                            size_t nextPos;
                            if (c < numCols -1)
                                nextPos = static_cast<size_t>(posMap[r].second[c+1]); // skip first offset being 0
                            else if (r < numRows - 1) // last column 
                                nextPos = static_cast<size_t>(posMap[r+1].first) - baseOffset; // first position of next row
                            else // last element
                                nextPos = fileBuffer.size() - baseOffset; // end of file
                            if (opt.useDoubleQuoteEncode){
                                std::string val ="";
                                setCString(linePtr + pos, pos, &val, delim, nextPos - 1);
                                std::cout <<"val: " << val << std::endl;
                                reinterpret_cast<std::string *>(rawCols[c])[r] = val;
                                break;
                            }
                            
                            const char posChar =(linePtr + pos)[0] ;
                            const char nextPosChar = (linePtr + nextPos - 2)[0];
                            if ((nextPos - pos > 0) && posChar == '\"' && nextPosChar == '\"'){//remove quotes
                                pos +=1;
                                nextPos -= 1;
                            }
                            std::string val(linePtr + pos, nextPos - pos - 1);
                   
                            reinterpret_cast<std::string *>(rawCols[c])[r] = val;
                            break;
                        }
                        case ValueTypeCode::FIXEDSTR16: {
                            size_t nextPos;
                            if (c < numCols -1)
                                nextPos = static_cast<size_t>(posMap[r].second[c+1]); // skip first offset being 0
                            else if (r < numRows - 1) // last column 
                                nextPos = static_cast<size_t>(posMap[r+1].first) - baseOffset; // first position of next row
                            else // last element
                                nextPos = fileBuffer.size() - baseOffset; // end of file
                            const char posChar =(linePtr + pos)[0] ;
                            const char nextPosChar = (linePtr + nextPos - 2)[0];
                            if (posChar == '\"' && nextPosChar == '\"'){//remove quotes
                                pos +=1;
                                nextPos -= 1;
                            }
                            std::string val(linePtr + pos, nextPos - pos - 1);
                            if (opt.useDoubleQuoteEncode){
                                reinterpret_cast<std::string *>(rawCols[c])[r] = convertDoubleQuotes(val);
                            } else
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
                // std::cout << "read time: " << std::chrono::duration_cast<std::chrono::duration<double>>(clock::now()
                // - time).count() << std::endl;
                return;
            }
        }
        // Normal branch: iterate row by row and for each field save its absolute offset.
        std::vector<std::pair<std::streampos, std::vector<uint16_t>>> posMap;
        if (opt.opt_enabled && opt.posMap)
            posMap.resize(numRows);
        std::streampos currentPos = 0;
        for (size_t row = 0; row < numRows; row++) {
            ssize_t ret = getFileLine(file);
            if ((file->read == EOF) || (file->line == NULL))
                break;
            if (ret == -1)
                throw std::runtime_error("ReadCsvFile::apply: getFileLine failed");

            // Save absolute offset for this row.
            if(opt.opt_enabled && opt.posMap) {
                posMap[row].first = currentPos;
                posMap[row].second.push_back(static_cast<uint16_t >(0));
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
                
                if (opt.opt_enabled && opt.posMap) {
                    if (col < numCols - 1) {
                        if (offset > 0) {
                            posMap[row].second.push_back(
                                static_cast<uint16_t>(pos + offset)); // adds offset from possible multiline string
                        } else
                            posMap[row].second.push_back(static_cast<uint16_t>(pos));
                    }
                }
                
            }
            currentPos = file->pos;
        }
        // std::cout << "read time: " << std::chrono::duration_cast<std::chrono::duration<double>>(clock::now() -
        // time).count() << std::endl;

        if (opt.opt_enabled) {
            if (opt.posMap) {
                try {
                    // auto writeTime = clock::now();
                    writePositionalMap(filename, posMap);
                    // std::cout<< "write time: "<<
                    // std::chrono::duration_cast<std::chrono::duration<double>>(clock::now() - writeTime).count() <<
                    // std::endl;

                } catch (std::exception &e) {
                    // positional map can still be used
                }
            }
        }
        delete[] rawCols;
        delete[] colTypes;
    }
};