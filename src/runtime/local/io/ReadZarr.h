/*
 * Copyright 2023 The DAPHNE Consortium
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

#ifndef ZARR_IO_H
#define ZARR_IO_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <optional>
#include <vector>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <utility>

#include <spdlog/spdlog.h>

#include <fcntl.h>

#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/ContiguousTensor.h>
#include <runtime/local/datastructures/ChunkedTensor.h>
#include <runtime/local/io/ZarrFileMetadata.h>
#include <runtime/local/io/ZarrUtils.h>
#include <parser/metadata/ZarrFileMetaDataParser.h>

enum struct IO_TYPE { POSIX, IO_URING };

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************
template<class DTRes>
struct ReadZarr {
    static void apply(DTRes *&res, const char *filename) = delete;
};

template<class DTRes>
struct PartialReadZarr {
    static void apply(DTRes *&res, const char *filename, const std::vector<std::pair<size_t, size_t>> &element_id_ranges) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************
template<class DTRes>
void readZarr(DTRes *&res, const char *filename) {
    auto fmd = ZarrFileMetaDataParser::readMetaData(filename);
    std::shared_ptr<spdlog::logger> log = GetZarrLogger();
    ReadZarr<DTRes>::apply(res, filename, fmd, log);
}

template<class DTRes>
void readZarr(DTRes *&res, const char *filename, const std::vector<std::pair<size_t, size_t>> &element_id_ranges) {
    auto fmd = ZarrFileMetaDataParser::readMetaData(filename);
    std::shared_ptr<spdlog::logger> log = GetZarrLogger();

    std::stringstream ss;
    for (const auto& e : element_id_ranges) {
        ss << "(" << e.first << ";" << e.second << ")";
    }
    // log->info("Issuing a partial read on file '{}' for element ranges [{}]", filename, ss.str());
    PartialReadZarr<DTRes>::apply(res, filename, fmd, element_id_ranges, log);
}

// template<class DTRes>
// void partialReadZarr(DTRes *&res, const char *filename, uint32_t lowerX, uint32_t upperX, uint32_t lowerY, uint32_t upperY, uint32_t lowerZ, uint32_t upperZ) {
//     auto fmd = ZarrFileMetaDataParser::readMetaData(filename);
//     std::shared_ptr<spdlog::logger> log = GetZarrLogger();
//
//     std::stringstream ss;
//     ss << "(" << lowerX << "," << upperX << ")";
//     ss << "(" << lowerY << "," << upperY << ")";
//     ss << "(" << lowerZ << "," << upperZ << ")";
//     // log->debug("Issuing a partial read on file '{}' for element ranges [{}]", filename, ss.str());
//     PartialReadZarr<DTRes>::apply(res, filename, fmd, {{lowerX, upperX},{lowerY,upperY},{lowerZ,upperZ}}, log);
// }

template<typename VT>
void CheckZarrMetaDataVT(ZarrDatatype read_type) {
    switch (read_type) {
        using enum ZarrDatatype;
        case BOOLEAN:
            if (!std::is_same<bool, VT>::value) {
                throw std::runtime_error("ReadZarr: read VT is bool != exptected VT");
            }
            break;
        case FP64:
            if (!std::is_same<double, VT>::value) {
                throw std::runtime_error("ReadZarr: read VT is double != exptected VT");
            }
            break;
        case FP32:
            if (!std::is_same<float, VT>::value) {
                throw std::runtime_error("ReadZarr: read VT is float != exptected VT");
            }
            break;
        case UINT64:
            if (!std::is_same<uint64_t, VT>::value) {
                throw std::runtime_error("ReadZarr: read VT is uint64_t != exptected VT");
            }
            break;
        case UINT32:
            if (!std::is_same<uint32_t, VT>::value) {
                throw std::runtime_error("ReadZarr: read VT is uint32_t != exptected VT");
            }
            break;
        case UINT16:
            if (!std::is_same<uint16_t, VT>::value) {
                throw std::runtime_error("ReadZarr: read VT is uint16_t != exptected VT");
            }
            break;
        case UINT8:
            if (!std::is_same<uint8_t, VT>::value) {
                throw std::runtime_error("ReadZarr: read VT is uint8_t != exptected VT");
            }
            break;
        case INT64:
            if (!std::is_same<int64_t, VT>::value) {
                throw std::runtime_error("ReadZarr: read VT is int64_t != exptected VT");
            }
            break;
        case INT32:
            if (!std::is_same<int32_t, VT>::value) {
                throw std::runtime_error("ReadZarr: read VT is int32_t != exptected VT");
            }
            break;
        case INT16:
            if (!std::is_same<int16_t, VT>::value) {
                throw std::runtime_error("ReadZarr: read VT is int16_t != exptected VT");
            }
            break;
        case INT8:
            if (!std::is_same<int8_t, VT>::value) {
                throw std::runtime_error("ReadZarr: read VT is int8_t != exptected VT");
            }
            break;
        default:
            throw std::runtime_error("ReadZarr: read VT currently not supported");
            break;
    }
}

template<typename VT>
void ReadChunk(const std::string& chunk_file_path, const std::vector<uint64_t>& dest_chunk_ids, uint64_t chunk_size_in_bytes, ChunkedTensor<VT>* dest, bool endian_missmatch, std::shared_ptr<spdlog::logger> lgr) {
    std::ifstream f;
    f.open(chunk_file_path, std::ios::in | std::ios::binary);

    if (!f.good() || !f.is_open()) {
        lgr->error("ReadZarr->ChunkedTensor: failed to open chunk file '{}' ", chunk_file_path);
        throw std::runtime_error("ReadZarr->ChunkedTensor: failed to open chunk file.");
    }

    f.read(reinterpret_cast<char *>(dest->getPtrToChunk(dest_chunk_ids)), chunk_size_in_bytes);

    if (!f.good()) {
        lgr->error("ReadZarr->ChunkedTensor: failed to read chunk file '{}'", chunk_file_path);
        throw std::runtime_error("ReadZarr->ChunkedTensor: failed to read chunk file.");
    }

    // Files endianness does not match the native endianness -> byte reverse every read element in the read chunk
    if (endian_missmatch) {
        ReverseArray(dest->data.get(), chunk_size_in_bytes / sizeof(VT));
    }

    dest->chunk_materialization_flags[dest->getLinearChunkIdFromChunkIds(dest_chunk_ids)] = true;
}

// Reads an entire tensor contained in a zarr "file/archive" into a chunked tensor. Both the tensor_shape and chunk_shape
// are determined by the zarr file. For reading only a portion see the partialRead functions.
template<typename VT>
struct ReadZarr<ChunkedTensor<VT>> {
    static void apply(ChunkedTensor<VT> *&res, const char *filename, const ZarrFileMetaData& zfmd, std::shared_ptr<spdlog::logger> log) {
        CheckZarrMetaDataVT<VT>(zfmd.data_type);

        std::vector<size_t> chunks_per_dim = computeChunksPerDim(zfmd.chunks, zfmd.shape);
        uint64_t elements_per_chunk = computeElementsPerChunk(zfmd.chunks, chunks_per_dim.size());

        // Fetch all available chunk keys within the respective directory
        std::string base_file_path                                         = filename;
        std::vector<std::pair<std::string, std::string>> chunk_keys_in_dir = GetAllChunkKeys(base_file_path);

        // Check retrieved keys for validity and generate full canonical path and associated chunk ids from it
        std::vector<std::vector<size_t>> chunk_ids;
        std::vector<std::string> full_chunk_file_paths;
        for (size_t i = 0; i < chunk_keys_in_dir.size(); i++) {
            auto tmp = GetChunkIdsFromChunkKey(std::get<1>(chunk_keys_in_dir[i]), zfmd.dimension_separator, zfmd.shape, chunks_per_dim);

            if (tmp) {
                full_chunk_file_paths.push_back(std::filesystem::canonical(std::filesystem::path(std::get<0>(chunk_keys_in_dir[i]))));
                chunk_ids.push_back(tmp.value());
            }
        }

        // log->info("Reading {} out of {} chunks", chunk_ids.size(), chunk_ids.size());

        res = DataObjectFactory::create<ChunkedTensor<VT>>(zfmd.shape, zfmd.chunks, InitCode::NONE);

        uint64_t chunk_size_in_bytes = elements_per_chunk * sizeof(VT);
        bool endianness_match = checkEndiannessMatch(zfmd.byte_order, log);

        // For all requested chunks open the respective file, fetch the ptr to the chunks location in the chunked
        // tensor and directly read into it
        for (size_t i = 0; i < full_chunk_file_paths.size(); i++) {
            ReadChunk<VT>(full_chunk_file_paths[i], chunk_ids[i], chunk_size_in_bytes, res, !endianness_match, log);
        }
    }
};

// Reads a full Zarr "file/archive" into a contiguous tensor.
// Unless the file in question contains a tensor with only one chunk, this process will involve actually 
// rechunking (i.e. it is not a noop in that case).
template<typename VT>
struct ReadZarr<ContiguousTensor<VT>> {
    static void apply(ContiguousTensor<VT> *&res, const char *filename, const ZarrFileMetaData& zfmd, std::shared_ptr<spdlog::logger> log) {
        ChunkedTensor<VT>* intermediate_tensor = nullptr;

        ReadZarr<ChunkedTensor<VT>>::apply(intermediate_tensor, filename, zfmd, log);

        if (!intermediate_tensor->tryRechunk(intermediate_tensor->tensor_shape)) {
            throw std::runtime_error("ReadZarr->ContiguosTensor: failed to rechunk the chunked tensor read from file to the requested contiguous layout");
        }

        res = DataObjectFactory::create<ContiguousTensor<VT>>(intermediate_tensor->data, intermediate_tensor->tensor_shape);

        DataObjectFactory::destroy(intermediate_tensor);
    }
};


enum struct ChunkAlignment {
    All_chunks_fully_alinged,
    Only_right_side_trunked,
    Has_left_side_trunkated,
};
    
ChunkAlignment CheckAlignment(const std::vector<uint64_t>& chunk_shape, const std::vector<std::pair<uint64_t,uint64_t>>& element_ranges) {
    for(size_t i=0; i < chunk_shape.size(); i++) {
        if (std::get<0>(element_ranges[i]) % chunk_shape[i] != 0) {
            return ChunkAlignment::Has_left_side_trunkated;
        }
    }
    for(size_t i=0; i < chunk_shape.size(); i++) {
        if (std::get<1>(element_ranges[i]) % chunk_shape[i] != 0) {
            return ChunkAlignment::Only_right_side_trunked;
        }
    }
    return ChunkAlignment::All_chunks_fully_alinged;
}

// Read a part of a tensor in a zarr "file/archive" into a chunked tensor. The section to be read is indicated by the
// element_id_ranges parameter.
// The chunk_shape specified in the zarr file will always also be the chunk shape of the resulting tensor.
// The resulting tensor will have the shape implied by the range, e.g. reading from a tensor with tensor_shape = {20,20}
// and chunk_shape {10,10}, with the ranges [0,10),[10,20) will result in a chunked tensor with tensor_shape = {10,10}
// and chunk_shape = {10,10}
// Id ranges that do not match up with the boundaries of a chunk will always result in all resulting partial chunks still
// being read completely from file. The resulting tensor will be, similar to the example above trunked and or diced to
// the requested tensor_shape. If the id range does not align on the right boundary (i.e. a partial overhanging chunk
// this does not further costs beyond the "full" read from file for that partial chunk. 
// Example: Same file as above. Id range [0,12), [10,12) -> resulting tensor_shape {12,2}, but still with chunking {10,10}
// If the range does not match the left boundary the tensor is diced to fit the requested range, which in this case is
// a expensive change to the memory layout.
// Example: Same file as above. Id range [1,13), [11,13) -> also results in tensor_shape {12,2}, but of course with
// different elements and notably requires the expensive dice operation mentioned above
// Todo: change this to immediately use a tensor of the correct size rather then using a "full" sized tensor followed by
// a dice to avoid the memcpies and allocs
template<typename VT>
struct PartialReadZarr<ChunkedTensor<VT>> {
    static void apply(ChunkedTensor<VT> *&res, const char *filename, const ZarrFileMetaData& zfmd, const std::vector<std::pair<size_t, size_t>> &element_id_ranges, std::shared_ptr<spdlog::logger> log) {
        ChunkedTensor<VT>* initial_tensor = nullptr;

        partialReadZarrTouchedChunks<VT>(initial_tensor, filename, zfmd, element_id_ranges, log);

        ChunkAlignment chunk_alignment = CheckAlignment(zfmd.chunks, element_id_ranges);

        switch (chunk_alignment) {
            using enum ChunkAlignment;

            case All_chunks_fully_alinged:
                {
                res = initial_tensor->tryDiceAtChunkLvl(initial_tensor->GetChunkRangeFromIdRange(element_id_ranges).value());
                if (res == nullptr) {
                    throw std::runtime_error("PartialReadZarr->ChunkedTensor: Dice at chunk lvl failed");
                }
                break;
                }
            case Only_right_side_trunked:
                {
                res = initial_tensor->tryDiceAtChunkLvl(initial_tensor->GetChunkRangeFromIdRange(element_id_ranges).value());
                if (res == nullptr) {
                    throw std::runtime_error("PartialReadZarr->ChunkedTensor: Dice at chunk lvl failed");
                }

                // We can simply adjust the tensor shape here as the memory layout of a tensor with and without partial
                // overhanging chunks is the same.
                std::vector<size_t> new_tensor_shape;
                size_t new_total_element_count = 1;
                for (size_t i=0; i < initial_tensor->rank; i++) {
                    new_tensor_shape.push_back(std::get<1>(element_id_ranges[i]) - std::get<0>(element_id_ranges[i]));
                    new_total_element_count *= new_tensor_shape.back();
                }
                res->tensor_shape = new_tensor_shape;
                res->total_element_count = new_total_element_count;

                break;
                }
            case Has_left_side_trunkated:
                {
                res = initial_tensor->tryDice(element_id_ranges, initial_tensor->chunk_shape);
                if (res == nullptr) {
                    throw std::runtime_error("PartialReadZarr->ChunkedTensor: Dice to final shape failed");
                }

                break;
                }
        }

        DataObjectFactory::destroy(initial_tensor);
    }
};

// Like PartialReadZarr<ChunkedTensor> but produces a contiguous tensor. If only a single chunk is read no changes to
// the memory layout will be performed.
template<typename VT>
struct PartialReadZarr<ContiguousTensor<VT>> {
    static void apply(ContiguousTensor<VT> *&res,
                      const char *filename,
                      const ZarrFileMetaData& zfmd,
                      const std::vector<std::pair<size_t, size_t>> &element_id_ranges,
                      std::shared_ptr<spdlog::logger> log) {
        ChunkedTensor<VT>* initial_tensor = nullptr;

        partialReadZarrTouchedChunks(initial_tensor, filename, zfmd, element_id_ranges, log);

        ChunkAlignment chunk_alignment = CheckAlignment(zfmd.chunks, element_id_ranges);

        auto required_chunk_list = initial_tensor->GetChunkListFromIdRange(element_id_ranges);

        if (required_chunk_list.value().size() == 1 && chunk_alignment == ChunkAlignment::All_chunks_fully_alinged) {
            res = DataObjectFactory::create<ContiguousTensor<VT>>(initial_tensor->getPtrToChunk(required_chunk_list.value()[0]), initial_tensor->chunk_shape);
        } else {
            res = initial_tensor->tryDiceToContiguousTensor(element_id_ranges);
            if (res == nullptr) {
                throw std::runtime_error("PartialReadZarr->ContiguousTensor: Dice to final shape failed");
            }
        }

        DataObjectFactory::destroy(initial_tensor);
    }
};

template<typename VT>
void partialReadZarrTouchedChunks(ChunkedTensor<VT> *&res, const char *filename, const ZarrFileMetaData& zfmd, const std::vector<std::pair<size_t, size_t>> &element_id_range, std::shared_ptr<spdlog::logger> log) {
    if (element_id_range.size() != zfmd.shape.size()) {
        throw std::runtime_error("PartialReadZarr->ChunkedTensor: Number of chunk ranges does not match tensor dim");
     }

    CheckZarrMetaDataVT<VT>(zfmd.data_type);

    res = DataObjectFactory::create<ChunkedTensor<VT>>(zfmd.shape, zfmd.chunks, InitCode::NONE);

    // Convert element ranges into list of chunks required
    std::optional<std::vector<std::vector<size_t>>> requested_chunk_ids = res->GetChunkListFromIdRange(element_id_range);

    if (!requested_chunk_ids) {
        throw std::runtime_error("PartialReadZarr->ChunkedTensor: Invalid element range. Range out-of-bounds or has mismatching dimension");
    }

    std::vector<size_t> chunks_per_dim;
    chunks_per_dim.resize(zfmd.shape.size());
    // uint64_t total_number_of_chunks = 1;
    for (size_t i = 0; i < chunks_per_dim.size(); i++) {
        chunks_per_dim[i] = zfmd.shape[i] / zfmd.chunks[i];
        // total_number_of_chunks *= chunks_per_dim[i];
    }
    uint64_t elements_per_chunk = computeElementsPerChunk(zfmd.chunks, chunks_per_dim.size());

    // Fetch all available chunk keys within the respective directory
    std::string base_file_path                                         = filename;
    std::vector<std::pair<std::string, std::string>> chunk_keys_in_dir = GetAllChunkKeys(base_file_path);

    // Check retrieved keys for validity and generate full canonical path and associated chunk ids from it
    std::vector<std::vector<size_t>> chunk_ids;
    std::vector<std::string> full_chunk_file_paths;
    for (size_t i = 0; i < chunk_keys_in_dir.size(); i++) {
        auto tmp = GetChunkIdsFromChunkKey(std::get<1>(chunk_keys_in_dir[i]), zfmd.dimension_separator, zfmd.shape, chunks_per_dim);

        if (tmp) {
            full_chunk_file_paths.push_back(std::filesystem::canonical(std::filesystem::path(std::get<0>(chunk_keys_in_dir[i]))));
            chunk_ids.push_back(tmp.value());
        }
    }

    // std::stringstream ss;
    // for (const auto& e : requested_chunk_ids.value()) {
    //     ss << "[";
    //     for (const auto& e1 : e) {
    //         ss << e1 << ", ";
    //     }
    //     ss << "]";
    // }

    // log->debug("Reading {} out of {} chunks", requested_chunk_ids.value().size(), total_number_of_chunks);

    // Match requested chunks to the available chunks in the fs, discard not-requested files and throw on missing file
    std::vector<std::string> full_requested_chunk_file_paths = computeFullFilePathsForRequestedChunks(requested_chunk_ids.value(), full_chunk_file_paths, chunk_ids);

    bool endianness_match = checkEndiannessMatch(zfmd.byte_order, log);
    uint64_t chunk_size_in_bytes = elements_per_chunk * sizeof(VT);

    // For all requested chunks open the respective file, fetch the ptr to the chunks location in the chunked
    // tensor and directly read into it
    for (size_t i = 0; i < full_requested_chunk_file_paths.size(); i++) {
        ReadChunk<VT>(full_requested_chunk_file_paths[i], requested_chunk_ids.value()[i], chunk_size_in_bytes, res, !endianness_match, log);
    }
}
#endif    // ZARR_IO_H
