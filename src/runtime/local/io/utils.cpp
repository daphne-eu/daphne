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

// Function to infer the data type of string value
ValueTypeCode inferValueType(const std::string &valueStr) {
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
FileMetaData generateFileMetaData(const std::string &filename, bool hasLabels, bool isFrame) {
    std::ifstream file(filename);
    std::string line;
    std::vector<ValueTypeCode> schema;
    std::vector<std::string> labels;
    size_t numRows = 0;
    size_t numCols = 0;
    bool isSingleValueType = false;
    // set the default value type to the most specific value type
    ValueTypeCode maxValueType = ValueTypeCode::SI8;
    ValueTypeCode currentType = ValueTypeCode::INVALID;

    if (file.is_open()) {
        if (isFrame) {
            if (hasLabels) {
                // extract labels from first line
                if (std::getline(file, line)) {
                    std::stringstream ss(line);
                    std::string label;
                    while (std::getline(ss, label, ',')) {
                        // trim any whitespaces for last element in line
                        //  Remove any newline characters from the end of the value
                        if (!label.empty() && (label.back() == '\n' || label.back() == '\r')) {
                            label.pop_back();
                        }
                        labels.push_back(label);
                    }
                }
            }
            // Read the rest of the file to infer the schema
            while (std::getline(file, line)) {
                std::stringstream ss(line);
                std::string value;
                size_t colIndex = 0;
                while (std::getline(ss, value, ',')) {
                    // trim any whitespaces for last element in line
                    //  Remove any newline characters from the end of the value
                    if (!value.empty() && (value.back() == '\n' || value.back() == '\r')) {
                        value.pop_back();
                    }
                    ValueTypeCode inferredType = inferValueType(value);
                    std::cout << "inferred valueType: " << static_cast<int>(inferredType) << ", " << value << "."
                              << std::endl;
                    // fill empty schema with inferred type
                    if (numCols <= colIndex) {
                        schema.push_back(inferredType);
                    }
                    currentType = schema[colIndex];
                    // update the current type if the inferred type is more specific
                    if (generality(currentType) < generality(inferredType)) {
                        currentType = inferredType;
                        schema[colIndex] = currentType;
                    }
                    if (generality(maxValueType) < generality(currentType)) {
                        maxValueType = currentType;
                    }
                    colIndex++;
                }
                numCols = std::max(numCols, colIndex);
                numRows++;
            }
            file.close();
        } else { // matrix
            while (std::getline(file, line)) {
                std::stringstream ss(line);
                std::string value;
                size_t colIndex = 0;
                while (std::getline(ss, value, ',')) {
                    if (!value.empty() && (value.back() == '\n' || value.back() == '\r')) {
                        value.pop_back();
                    }
                    ValueTypeCode inferredType = inferValueType(value);
                    if (generality(maxValueType) < generality(inferredType)) {
                        maxValueType = inferredType;
                    }
                    colIndex++;
                }
                numCols = std::max(numCols, colIndex);
                numRows++;
            }
            schema.clear();
            schema.push_back(maxValueType);
            isSingleValueType = true;
        }
        file.close();
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }

    return FileMetaData(numRows, numCols, isSingleValueType, schema, labels);
}