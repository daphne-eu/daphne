/*
 * Copyright 2024 The DAPHNE Consortium
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

#include <optional>
#include <vector>
#include <string>
#include <utility>
#include <cstdint>

enum class ByteOrder { LITTLEENDIAN, BIGENDIAN, NOT_RELEVANT };

enum class ZarrDatatype {
    BOOLEAN,
    FP64,
    FP32,
    UINT64,
    UINT32,
    UINT16,
    UINT8,
    INT64,
    INT32,
    INT16,
    INT8,
    COMPLEX_FLOATING,
    TIMEDELTA,
    DATETIME,
    STRING,
    UNICODE,
    OTHER
};

struct ZarrFileMetaData {
    std::vector<size_t> chunks;
    std::vector<size_t> shape;
    uint16_t zarr_format;
    std::string order;
    std::string fill_value;
    std::string dtype;
    std::string dimension_separator = ".";
    ByteOrder byte_order = ByteOrder::LITTLEENDIAN;
    ZarrDatatype data_type = ZarrDatatype::INT64;
    uint16_t nBytes = 8;
};

std::ostream& operator<<(std::ostream& out, const ByteOrder& bo);
std::ostream& operator<<(std::ostream& out, const ZarrDatatype& dt);
std::ostream& operator<<(std::ostream& out, ZarrFileMetaData& zm);

std::optional<std::vector<size_t>> GetChunkIdsFromChunkKey(const std::string &chunk_key_to_test,
                                                           const std::string &dim_separator,
                                                           const std::vector<size_t> &tensor_shape,
                                                           const std::vector<size_t> &amount_of_chunks_per_dim);

std::vector<std::pair<std::string, std::string>> GetAllChunkKeys(const std::string &base_dir_file_path);
