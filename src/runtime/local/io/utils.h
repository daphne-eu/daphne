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

#include <spdlog/spdlog.h>

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
  }
  catch (const std::invalid_argument &) {
    *v = std::numeric_limits<float>::quiet_NaN();
  }
  catch (const std::out_of_range& e) {
      // handling subnormal values (too small)
      *v = std::numeric_limits<float>::min();
      spdlog::warn("setting subnormal float value {} to std::numeric_limits<float>::min() -> {}", x, std::numeric_limits<float>::min());
  }
}
inline void convertStr(std::string const &x, int8_t *v) { *v = stoi(x); }
inline void convertStr(std::string const &x, int32_t *v) { *v = stoi(x); }
inline void convertStr(std::string const &x, int64_t *v) { *v = stoi(x); }
inline void convertStr(std::string const &x, uint8_t *v) { *v = stoi(x); }
inline void convertStr(std::string const &x, uint32_t *v) { *v = stoi(x); }
inline void convertStr(std::string const &x, uint64_t *v) { *v = stoi(x); }

// Conversion of char *.

inline void convertCstr(const char * x, double *v) {
  char * end;
  *v = strtod(x, &end);
  if(x == end)
    *v = std::numeric_limits<double>::quiet_NaN();
}
inline void convertCstr(const char * x, float *v) {
  char * end;
  *v = strtof(x, &end);
  if(x == end)
    *v = std::numeric_limits<float>::quiet_NaN();
}
inline void convertCstr(const char * x, int8_t *v) { *v = atoi(x); }
inline void convertCstr(const char * x, int32_t *v) { *v = atoi(x); }
inline void convertCstr(const char * x, int64_t *v) { *v = atoi(x); }
inline void convertCstr(const char * x, uint8_t *v) { *v = atoi(x); }
inline void convertCstr(const char * x, uint32_t *v) { *v = atoi(x); }
inline void convertCstr(const char * x, uint64_t *v) { *v = atoi(x); }


inline size_t setCString(const char * str, std::string *res, const char delim){
    size_t pos = 0;
    bool is_multiLine = (str[0] == '"');
    if(is_multiLine)
      pos++;

    int is_not_end = 1;
    while (is_not_end)
    {
      // The string does not contain line breaks or a field separator, so
      // the end of the string is either a delimiter or the next character is the end of the line.
      is_not_end -= (!is_multiLine && str[pos] == delim);
      is_not_end -= (!is_multiLine && str[pos + 1] == '\n');

      /*
      ** If the string contains line breaks or field separators, 
      ** it must be enclosed in double quotes. We then skip all
      ** characters until we find the closing double quote.
      ** If a double quote appears inside the string, it must be escaped 
      ** by doubling the double quote (""), or by preceding it with a backslash (\).
      */
      is_not_end -= (is_multiLine && str[pos] == '"' && str[pos + 1] != '"');
      if (!is_not_end)
        break;
      pos += (is_multiLine && str[pos] == '"' && str[pos + 1] == '"');
      pos += (is_multiLine && str[pos] == '\\' && str[pos + 1] == '"');

      pos++;
    }

    if(is_multiLine)
      res->append(str + 1, pos - 1);
    else
      res->append(str, pos);
    
    if(is_multiLine)
      pos++;

    // The result `pos` should point to the character just before the next column.
    return pos;
}
