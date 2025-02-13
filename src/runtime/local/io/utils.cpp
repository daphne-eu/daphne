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

#include <fstream>
#include <iostream>
#include <limits>
#include <runtime/local/datastructures/FixedSizeStringValueType.h>
#include <runtime/local/io/FileMetaData.h>
#include <runtime/local/io/utils.h>
#include <sstream>
#include <string>
#include <vector>

int generality(ValueTypeCode type) { // similar to generality in TypeInferenceUtils.cpp but for ValueTypeCode
    switch (type) {
    case ValueTypeCode::SI8:
        return 0;
    case ValueTypeCode::UI8:
        return 1;
    case ValueTypeCode::SI32:
        return 2;
    case ValueTypeCode::UI32:
        return 3;
    case ValueTypeCode::SI64:
        return 4;
    case ValueTypeCode::UI64:
        return 5;
    case ValueTypeCode::F32:
        return 6;
    case ValueTypeCode::F64:
        return 7;
    case ValueTypeCode::FIXEDSTR16:
        return 8;
    default:
        return 9;
    }
}

ValueTypeCode inferValueType(const char* line, size_t &pos, char delim) {
    std::string field;
    // Extract field until delimiter
    while (line[pos] != delim && line[pos] != '\0') {
        field.push_back(line[pos]);
        pos++;
    }
    // Skip delimiter if present.
    if (line[pos] == delim)
        pos++;
    return inferValueType(field);
}

// Function to infer the data type of string value
ValueTypeCode inferValueType(const std::string &valueStr) {
    std::cout << "inferred value:" << valueStr <<";"<< std::endl;
    // Check if the string represents an integer
    bool isInteger = true;
    for (char c : valueStr) {
        if (!isdigit(c) && c != '-' && c != '+' && c != ' ') {
            isInteger = false;
            break;
        }
    }

    if (isInteger) {
        try {
            int64_t value = std::stoll(valueStr);
            if (value >= std::numeric_limits<int8_t>::min() && value <= std::numeric_limits<int8_t>::max()) {
                return ValueTypeCode::SI8;
            } else if (value >= 0 && value <= std::numeric_limits<uint8_t>::max()) {
                return ValueTypeCode::UI8;
            } else if (value >= std::numeric_limits<int32_t>::min() && value <= std::numeric_limits<int32_t>::max()) {
                return ValueTypeCode::SI32;
            } else if (value >= 0 && value <= std::numeric_limits<uint32_t>::max()) {
                return ValueTypeCode::UI32;
            } else if (value >= std::numeric_limits<int64_t>::min() && value <= std::numeric_limits<int64_t>::max()) {
                return ValueTypeCode::SI64;
            } else {
                return ValueTypeCode::UI64;
            }
        } catch (const std::invalid_argument &) {
            // Continue to next check
        } catch (const std::out_of_range &) {
            return ValueTypeCode::UI64;
        }
    }

    // Check if the string represents a float
    try {
        float fvalue = std::stof(valueStr);
        if (fvalue >= std::numeric_limits<float>::lowest() && fvalue <= std::numeric_limits<float>::max()) {
            return ValueTypeCode::F32;
        }
    } catch (const std::invalid_argument &) {
        // Continue to next check
    } catch (const std::out_of_range &) {
        // Continue to next check
    }

    // Check if the string represents a double
    try {
        double dvalue = std::stod(valueStr);
        if (dvalue >= std::numeric_limits<double>::lowest() && dvalue <= std::numeric_limits<double>::max()) {
            return ValueTypeCode::F64;
        }
    } catch (const std::invalid_argument &) {
        // Continue to next check
    } catch (const std::out_of_range &) {
        // Continue to next check
    }

    if (valueStr.size() == 16) {
        return ValueTypeCode::FIXEDSTR16;
    }
    return ValueTypeCode::STR;
}

// Function to read the CSV file and determine the FileMetaData
FileMetaData generateFileMetaData(const std::string &filename, char delim, size_t sampleRows) {
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Cannot open file: " + filename);
    std::cout << "Reading file: " << filename << std::endl;
    std::string line;
    std::vector<ValueTypeCode> colTypes; // will be resized once we know numCols
    bool firstLine = true;
    size_t row = 0;
    while (std::getline(file, line) && row < sampleRows) {
        size_t pos = 0;
        size_t col = 0;
        // On first row, determine number of columns.
        if (firstLine) {
            // Count the number of delimiters + 1
            size_t ncols = 1;
            for (char c : line)
                if (c == delim)
                    ncols++;
            colTypes.resize(ncols, ValueTypeCode::SI8); // start with narrow type.
            firstLine = false;
        }
        // Process each token.
        while (pos < line.size() && col < colTypes.size()) {
            size_t tempPos = pos;
            ValueTypeCode tokenType = inferValueType(line.c_str(), tempPos, delim);
            std::cout << "inferred Token type: " << static_cast<int>(tokenType) << std::endl;
            // Promote type if needed.
            if (generality(tokenType) > generality(colTypes[col]))
                colTypes[col] = tokenType;
            pos = tempPos;
            col++;
        }
        row++;
    }
    file.close();
    std::vector<std::string> labels;
    size_t numCols=colTypes.size();
    bool isSingleValueType = true;
    ValueTypeCode maxValueType = colTypes[0];
    for (size_t i = 0; i < numCols; i++) {
        labels.push_back("col_" + std::to_string(i));
        if (maxValueType != colTypes[i]) {
            isSingleValueType = false;
        }
    }
    FileMetaData fmd = FileMetaData(row, colTypes.size(), isSingleValueType, colTypes, labels);
    return fmd;
}

void readCsvLine(File* file, size_t row, char delim, size_t numCols, uint8_t **rawCols, ValueTypeCode* colTypes, bool genTypes = true) {
    size_t pos, col = 0;
    ValueTypeCode colType = ValueTypeCode::INVALID;
        while (1) {
            if (colTypes != nullptr){
                colType = colTypes[col];
            }else if(!genTypes){
                throw std::runtime_error("ReadCsvFile::apply: colTypes must be provided if genTypes is false");
            }else{ // set colTypes to most specific value type possible
                colTypes= new ValueTypeCode[numCols];
                for (size_t i = 0; i < numCols; i++) {
                    colTypes[i] = ValueTypeCode::SI8;
                }
            }
            if (genTypes){
                colType = inferValueType(file->line, pos, delim);
                if (generality(colType) > generality(colTypes[col])) {
                    colTypes[col] = colType;
                }
            }
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
                pos = setCString(file, pos, &val_str, delim);
                reinterpret_cast<std::string *>(rawCols[col])[row] = val_str;
                break;
            }
            case ValueTypeCode::FIXEDSTR16: {
                std::string val_str = "";
                pos = setCString(file, pos, &val_str, delim);
                reinterpret_cast<FixedStr16 *>(rawCols[col])[row] = FixedStr16(val_str);
                break;
            }
            default:
                throw std::runtime_error("ReadCsvFile::apply: unknown value type code");
            }

            if (++col >= numCols) {
                break;
            }

            // TODO We could even exploit the fact that the strtoX functions
            // can return a pointer to the first character after the parsed
            // input, then we wouldn't have to search for that ourselves,
            // just would need to check if it is really the delimiter.
            while (file->line[pos] != delim)
                pos++;
            pos++; // skip delimiter
        }
}