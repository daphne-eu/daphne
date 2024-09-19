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
 
#include <runtime/local/io/ZarrFileMetaData.h>

#include <iostream>
#include <optional>
#include <vector>
#include <string>
#include <utility>
#include <cstdint>
#include <filesystem>
#include <algorithm>

std::ostream& operator<<(std::ostream& out, const ByteOrder& bo) {
    switch (bo) {
        case ByteOrder::LITTLEENDIAN:
            out << "\"little endian\"";
            break;
        case ByteOrder::BIGENDIAN:
            out << "\"big endian\"";
            break;
        case ByteOrder::NOT_RELEVANT:
            out << "\"not relevant\"";
            break;
    }
    return out;
}

std::ostream& operator<<(std::ostream& out, const ZarrDatatype& dt) {
    switch (dt) {
        using enum ZarrDatatype;
        case BOOLEAN:
            out << "\"boolean\"";
            break;
        case FP64:
            out << "\"double\"";
            break;
        case FP32:
            out << "\"float\"";
            break;
        case INT64:
            out << "\"int64_t\"";
            break;
        case INT32:
            out << "\"int32_t\"";
            break;
        case INT16:
            out << "\"int16_t\"";
            break;
        case INT8:
            out << "\"int8_t\"";
            break;
        case UINT64:
            out << "\"uint64_t\"";
            break;
        case UINT32:
            out << "\"uint32_t\"";
            break;
        case UINT16:
            out << "\"uint16_t\"";
            break;
        case UINT8:
            out << "\"uint8_t\"";
            break;
        case COMPLEX_FLOATING:
            out << "\"complex floating\"";
            break;
        case TIMEDELTA:
            out << "\"timedelta\"";
            break;
        case DATETIME:
            out << "\"datetime\"";
            break;
        case STRING:
            out << "\"string\"";
            break;
        case UNICODE:
            out << "\"unicode\"";
            break;
        case OTHER:
            out << "\"other\"";
            break;
    }
    return out;
}

std::ostream& operator<<(std::ostream& out, ZarrFileMetaData& zm) {
    out << "Chunks [";
    for (const auto& e : zm.chunks) {
        out << e << " ";
    }
    out << "]\n";
    out << "Shape [";
    for (const auto& e : zm.shape) {
        out << e << " ";
    }
    out << "]\n";
    out << "Zarr format: " << zm.zarr_format << "\n";
    out << "Order: " << zm.order << "\n";
    out << "Fill value: \"" << zm.fill_value << "\"\n";
    out << "Data type: " << zm.data_type << "\n";
    out << "Byte order: " << zm.byte_order << "\n";
    out << "#Bytes data type: " << zm.nBytes << "\n";
    out << "Dimension separator: " << zm.dimension_separator << "\n";
    return out;
}

enum struct ZarrParseCharState { IsNumeral, IsSeparator, IsInvalid };

std::optional<std::vector<size_t>> GetChunkIdsFromChunkKey(const std::string& chunk_key_to_test,
                                                           const std::string& dim_separator,
                                                           const std::vector<size_t>& tensor_shape,
                                                           const std::vector<size_t>& amount_of_chunks_per_dim) {
    std::vector<uint64_t> parsed_chunk_ids;
    std::string tmp;

    for (size_t i = 0; i < chunk_key_to_test.size(); i++) {
        ZarrParseCharState current_char_state;

        if (std::isdigit(chunk_key_to_test[i])) {
            current_char_state = ZarrParseCharState::IsNumeral;
        } else {
            if (chunk_key_to_test[i] == dim_separator[0]) {
                current_char_state = ZarrParseCharState::IsSeparator;
            } else {
                current_char_state = ZarrParseCharState::IsInvalid;
            }
        }

        int64_t chunk_id;
        switch (current_char_state) {
            using enum ZarrParseCharState;
            case IsNumeral:
                tmp += chunk_key_to_test[i];
                if (i == (chunk_key_to_test.size() - 1)) {
                    chunk_id = std::stol(tmp);

                    if ((chunk_id < 0) ||
                        chunk_id >= static_cast<int64_t>(amount_of_chunks_per_dim[parsed_chunk_ids.size()])) {
                        return std::nullopt;    // chunk id out of bounds
                    }

                    tmp.clear();
                    parsed_chunk_ids.push_back(static_cast<uint64_t>(chunk_id));
                }
                break;
            case IsSeparator:
                if (tmp.size() == 0) {
                    // encountered separator without preceding number
                    return std::nullopt;
                }
                chunk_id = std::stol(tmp);

                if ((chunk_id < 0) ||
                    chunk_id >= static_cast<int64_t>(amount_of_chunks_per_dim[parsed_chunk_ids.size()])) {
                    return std::nullopt;    // chunk id out of bounds
                }

                tmp.clear();
                parsed_chunk_ids.push_back(static_cast<uint64_t>(chunk_id));
                break;
            case IsInvalid:
                return std::nullopt;
        }
    }

    if (parsed_chunk_ids.size() != tensor_shape.size()) {
        return std::nullopt;
    }

    std::reverse(parsed_chunk_ids.begin(), parsed_chunk_ids.end());

    return parsed_chunk_ids;
}

std::string GetChunkKeyFromChunkIds(const std::string& dim_separator, const std::vector<size_t>& chunk_ids) {
    std::string key;
    for (size_t i=0; i<chunk_ids.size(); i++) {
        key += std::to_string(chunk_ids[i]);
        if (i != (chunk_ids.size() - 1)) {
            key += dim_separator;
        }
    }
    return key;
}

std::vector<std::pair<std::string, std::string>> GetAllChunkKeys(const std::string& base_dir_file_path) {
    std::vector<std::pair<std::string, std::string>> chunk_keys;

    for (auto const& dir_entry : std::filesystem::recursive_directory_iterator {base_dir_file_path}) {
        if (!dir_entry.is_directory()) {
            std::string full_path = dir_entry.path().string();
            size_t last_dir_sep   = full_path.find_last_of("/");
            chunk_keys.push_back({full_path, full_path.substr(last_dir_sep + 1)});
        }
    }

    return chunk_keys;
}
