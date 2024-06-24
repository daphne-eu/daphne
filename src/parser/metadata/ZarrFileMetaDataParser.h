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

#include <string>

struct ZarrFileMetaData;

struct ZarrFileMetaDataParser {
    static ZarrFileMetaData readMetaData(const std::string& filename);

    static const std::string ZARR_KEY_FILE_EXTENSION;
    static const std::string ZARR_KEY_CHUNKS;
    static const std::string ZARR_KEY_SHAPE;
    static const std::string ZARR_KEY_FORMAT;
    static const std::string ZARR_KEY_ORDER;
    static const std::string ZARR_KEY_FILLVALUE;
    static const std::string ZARR_KEY_DTYPE;
    static const std::string ZARR_KEY_DIMENSION_SEPARATOR;
    static const std::string ZARR_KEY_COMPRESSOR;
    static const std::string ZARR_KEY_FILTERS;
    static const char ZARR_KEY_VAL_BYTEORDER_LE;
    static const char ZARR_KEY_VAL_BYTEORDER_BE;
    static const char ZARR_KEY_VAL_BYTEORDER_NOT_RELEVANT;

    static const char ZARR_KEY_VAL_BYTELENGTH_1;
    static const char ZARR_KEY_VAL_BYTELENGTH_2;
    static const char ZARR_KEY_VAL_BYTELENGTH_4;
    static const char ZARR_KEY_VAL_BYTELENGTH_8;

    static const char ZARR_KEY_VAL_DTYPE_INTEGER;
    static const char ZARR_KEY_VAL_DTYPE_BOOLEAN;
    static const char ZARR_KEY_VAL_DTYPE_UNSIGNED_INTEGER;
    static const char ZARR_KEY_VAL_DTYPE_FLOAT;
    static const char ZARR_KEY_VAL_DTYPE_COMPLEX_FLOATING;
    static const char ZARR_KEY_VAL_DTYPE_TIMEDELTA;
    static const char ZARR_KEY_VAL_DTYPE_DATETIME;
    static const char ZARR_KEY_VAL_DTYPE_STRING;
    static const char ZARR_KEY_VAL_DTYPE_UNICODE;
    static const char ZARR_KEY_VAL_DTYPE_OTHER;
};
