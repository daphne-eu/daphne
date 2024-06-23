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

#include <vector>
#include <cstdint>
#include <type_traits>

#include <spdlog/spdlog.h>

#include <runtime/local/io/ZarrFileMetaData.h>

enum struct ChunkAlignment {
    All_chunks_fully_alinged,
    Only_right_side_trunked,
    Has_left_side_truncated,
};

bool checkEndiannessMatch(const ByteOrder bo, std::shared_ptr<spdlog::logger> log);

std::vector<size_t> computeChunksPerDim(const std::vector<size_t>& chunks, const std::vector<size_t>& shape);

uint64_t computeElementsPerChunk(const std::vector<size_t>& chunks, const size_t n);

std::vector<std::string> computeFullFilePathsForRequestedChunks(
    const std::vector<std::vector<size_t>>& requested_chunk_ids,
    std::vector<std::string>& full_chunk_file_paths,
    std::vector<std::vector<size_t>>& chunk_ids);

std::shared_ptr<spdlog::logger> GetZarrLogger();

ChunkAlignment CheckAlignment(const std::vector<uint64_t>& chunk_shape, const std::vector<std::pair<uint64_t,uint64_t>>& element_ranges);

template<typename T1, typename T2>
void validateDatatype(const std::string& datatype) {
    if (!std::is_same<T1, T2>::value) {
        throw std::runtime_error("ReadZarr: read VT is " + datatype + "!= exptected VT");
    }
}

template<typename VT>
void CheckZarrMetaDataVT(ZarrDatatype read_type) {
    switch (read_type) {
        using enum ZarrDatatype;
        case BOOLEAN:
            validateDatatype<bool, VT>("bool");
            break;
        case FP64:
            validateDatatype<double, VT>("double");
            break;
        case FP32:
            validateDatatype<float, VT>("float");
            break;
        case UINT64:
            validateDatatype<uint64_t, VT>("uint64_t");
            break;
        case UINT32:
            validateDatatype<uint32_t, VT>("uint32_t");
            break;
        case UINT16:
            validateDatatype<uint16_t, VT>("uint16_t");
            break;
        case UINT8:
            validateDatatype<uint8_t, VT>("uint8_t");
            break;
        case INT64:
            validateDatatype<int64_t, VT>("int64_t");
            break;
        case INT32:
            validateDatatype<int32_t, VT>("int32_t");
            break;
        case INT16:
            validateDatatype<int16_t, VT>("int16_t");
            break;
        case INT8:
            validateDatatype<int8_t, VT>("int8_t");
            break;
        default:
            throw std::runtime_error("ReadZarr: read VT currently not supported");
            break;
    }
}

