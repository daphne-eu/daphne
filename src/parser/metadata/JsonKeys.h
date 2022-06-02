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

#ifndef SRC_PARSER_METADATA_JSONPARAMS_H
#define SRC_PARSER_METADATA_JSONPARAMS_H

#include <string>

/**
 * @brief A container that contains names of JSON keys for a file
 * metadata.
 * 
 */
struct JsonKeys {

    // mandatory keys
    inline static const std::string NUM_ROWS = "numRows";   // int
    inline static const std::string NUM_COLS = "numCols";   // int

    // should always contain exactly one of the following keys
    inline static const std::string VALUE_TYPE = "valueType";   // string
    inline static const std::string SCHEMA = "schema";  // array of objects

    // optional key
    inline static const std::string NUM_NON_ZEROS = "numNonZeros";  // int (default: -1)
};

#endif