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

#include <limits>
#include <stdexcept>
#include <string>

#include <runtime/local/io/File.h>
#include <spdlog/spdlog.h>

#include <runtime/local/io/FileMetaData.h>

// Function to create and save the positional map
void writePositionalMap(const char *filename, const std::vector<std::vector<std::streampos>> &posMap);

// Function to read the positional map
std::vector<std::vector<std::streampos>> readPositionalMap(const char *filename, size_t numCols);

// Conversion of std::string.

inline void convertStr(std::string const &x, double *v) {
    try {
        *v = stod(x);
    } catch (const std::invalid_argument &) {
        *v = std::numeric_limits<double>::quiet_NaN();
    }
}
inline void convertStr(std::string const &x, float *v) {
    try {
        *v = stof(x);
    } catch (const std::invalid_argument &) {
        *v = std::numeric_limits<float>::quiet_NaN();
    } catch (const std::out_of_range &e) {
        // handling subnormal values (too small)
        *v = std::numeric_limits<float>::min();
        spdlog::warn("setting subnormal float value {} to "
                     "std::numeric_limits<float>::min() -> {}",
                     x, std::numeric_limits<float>::min());
    }
}
inline void convertStr(std::string const &x, int8_t *v) { *v = stoi(x); }
inline void convertStr(std::string const &x, int32_t *v) { *v = stoi(x); }
inline void convertStr(std::string const &x, int64_t *v) { *v = stoi(x); }
inline void convertStr(std::string const &x, uint8_t *v) { *v = stoi(x); }
inline void convertStr(std::string const &x, uint32_t *v) { *v = stoi(x); }
inline void convertStr(std::string const &x, uint64_t *v) { *v = stoi(x); }

// Conversion of char *.

inline void convertCstr(const char *x, double *v) {
    char *end;
    *v = strtod(x, &end);
    if (x == end)
        *v = std::numeric_limits<double>::quiet_NaN();
}
inline void convertCstr(const char *x, float *v) {
    char *end;
    *v = strtof(x, &end);
    if (x == end)
        *v = std::numeric_limits<float>::quiet_NaN();
}
inline void convertCstr(const char *x, int8_t *v) { *v = atoi(x); }
inline void convertCstr(const char *x, int32_t *v) { *v = atoi(x); }
inline void convertCstr(const char *x, int64_t *v) { *v = atoi(x); }
inline void convertCstr(const char *x, uint8_t *v) { *v = atoi(x); }
inline void convertCstr(const char *x, uint32_t *v) { *v = atoi(x); }
inline void convertCstr(const char *x, uint64_t *v) { *v = atoi(x); }

/**
 * @brief This function reads a CSV column that contains strings.
 *
 * This function processes a column from a CSV file starting at the given position in the current line.
 * It reads and appends characters to the result string (`res`) until it encounters the column delimiter
 * or the end of the line. If the column contains multiline strings (enclosed in double quotes), it
 * continues reading until the closing quote is found, handling embedded quotes and newline characters
 * as necessary.
 *
 * @param file Pointer to the file object from which the CSV data is being read. The file's `line`
 *             attribute is expected to contain the current line being processed.
 * @param start_pos The starting position within the current line to begin reading the column. This
 *                  function may move beyond the current line if the field contains a multiline string.
 * @param res A pointer to the result string that will store the contents of the current column.
 * @param delim The delimiter character separating columns (e.g., a comma `,`).
 * @return The position pointing to the character immediately before the next column in the line.
 */
inline size_t setCString(struct File *file, size_t start_pos, std::string *res, const char delim) {
    size_t pos = 0;
    const char *str = file->line + start_pos;
    bool is_multiLine = (str[0] == '"');
    if (is_multiLine)
        pos++;

    uint8_t has_line_break = 0;
    uint8_t is_not_end = 1;
    while (is_not_end && str[pos]) {
        is_not_end -= (!is_multiLine && str[pos] == delim);
        is_not_end -= (!is_multiLine && str[pos] == '\n');
        is_not_end -= (!is_multiLine && str[pos] == '\r');

        is_not_end -= (is_multiLine && str[pos] == '"' && str[pos + 1] != '"');
        if (!is_not_end)
            break;
        if (is_multiLine && str[pos] == '"' && str[pos + 1] == '"') {
            res->append("\"");
            pos += 2;
        } else if (is_multiLine && str[pos] == '\\' && str[pos + 1] == '"') {
            res->append("\\\"");
            pos += 2;
        } else if (is_multiLine && (str[pos] == '\n' || str[pos] == '\r')) {
            res->push_back('\n');
            getFileLine(file);
            str = file->line;
            pos = 0;
            has_line_break = 1;
        } else {
            res->push_back(str[pos]);
            pos++;
        }
    }

    if (is_multiLine)
        pos++;

    if (has_line_break)
        return pos;
    else
        return pos + start_pos;
}