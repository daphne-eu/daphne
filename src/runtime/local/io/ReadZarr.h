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

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <optional>
#include <vector>
#include <fstream>
#include <filesystem>
#include <bit>
#include <sstream>

#include <spdlog/spdlog.h>

#include <fcntl.h>
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
    std::shared_ptr<spdlog::logger> log = spdlog::get("runtime::io::zarr");
    
    if (fmd.shape.empty()) {
        log->error("Tensors of dimensionality 0 (i.e. scalars) are currently not supported during reading");
        throw std::runtime_error("Tensors of dim 0 i.e. scalars are currently not supported during reading");
    }

    ReadZarr<DTRes>::apply(res, filename, fmd, log);
}

template<class DTRes>
void readZarr(DTRes *&res, const char *filename, const std::vector<std::pair<size_t, size_t>> &element_id_ranges) {
    auto fmd = ZarrFileMetaDataParser::readMetaData(filename);
    std::shared_ptr<spdlog::logger> log = spdlog::get("runtime::io::zarr");

    if (fmd.shape.empty()) {
        log->error("Tensors of dimensionality 0 (i.e. scalars) are currently not supported during reading");
        throw std::runtime_error("Tensors of dim 0 i.e. scalars are currently not supported during reading");
    }
    std::stringstream ss;
    for (const auto& e : element_id_ranges) {
        ss << "(" << e.first << ";" << e.second << ")";
    }
    log->info("Issuing a partial read on file '{}' for element ranges [{}]", filename, ss.str());
    PartialReadZarr<DTRes>::apply(res, filename, fmd, element_id_ranges, log);
}

template<class DTRes>
void partialReadZarr(DTRes *&res, const char *filename, uint32_t lowerX, uint32_t upperX, uint32_t lowerY, uint32_t upperY, uint32_t lowerZ, uint32_t upperZ) {
    auto fmd = ZarrFileMetaDataParser::readMetaData(filename);
    std::shared_ptr<spdlog::logger> log = spdlog::get("runtime::io::zarr");

    if (fmd.shape.empty()) {
        log->error("Tensors of dimensionality 0 (i.e. scalars) are currently not supported during reading");
        throw std::runtime_error("Tensors of dim 0 i.e. scalars are currently not supported during reading");
    }
    std::stringstream ss;
    ss << "(" << lowerX << "," << upperX << ")";
    ss << "(" << lowerY << "," << upperY << ")";
    ss << "(" << lowerZ << "," << upperZ << ")";
    log->debug("Issuing a partial read on file '{}' for element ranges [{}]", filename, ss.str());
    PartialReadZarr<DTRes>::apply(res, filename, fmd, {{lowerX, upperX},{lowerY,upperY},{lowerZ,upperZ}}, log);
}

template<typename VT>
void CheckZarrMetaDataVT(ZarrDatatype read_type) {
    switch (read_type) {
        // using enum ZarrDatatype;
        case ZarrDatatype::BOOLEAN:
            if (!std::is_same<bool, VT>::value) {
                throw std::runtime_error("ReadZarr: read VT is bool != exptected VT");
            }
            break;
        case ZarrDatatype::FP64:
            if (!std::is_same<double, VT>::value) {
                throw std::runtime_error("ReadZarr: read VT is double != exptected VT");
            }
            break;
        case ZarrDatatype::FP32:
            if (!std::is_same<float, VT>::value) {
                throw std::runtime_error("ReadZarr: read VT is float != exptected VT");
            }
            break;
        case ZarrDatatype::UINT64:
            if (!std::is_same<uint64_t, VT>::value) {
                throw std::runtime_error("ReadZarr: read VT is uint64_t != exptected VT");
            }
            break;
        case ZarrDatatype::UINT32:
            if (!std::is_same<uint32_t, VT>::value) {
                throw std::runtime_error("ReadZarr: read VT is uint32_t != exptected VT");
            }
            break;
        case ZarrDatatype::UINT16:
            if (!std::is_same<uint16_t, VT>::value) {
                throw std::runtime_error("ReadZarr: read VT is uint16_t != exptected VT");
            }
            break;
        case ZarrDatatype::UINT8:
            if (!std::is_same<uint8_t, VT>::value) {
                throw std::runtime_error("ReadZarr: read VT is uint8_t != exptected VT");
            }
            break;
        case ZarrDatatype::INT64:
            if (!std::is_same<int64_t, VT>::value) {
                throw std::runtime_error("ReadZarr: read VT is int64_t != exptected VT");
            }
            break;
        case ZarrDatatype::INT32:
            if (!std::is_same<int32_t, VT>::value) {
                throw std::runtime_error("ReadZarr: read VT is int32_t != exptected VT");
            }
            break;
        case ZarrDatatype::INT16:
            if (!std::is_same<int16_t, VT>::value) {
                throw std::runtime_error("ReadZarr: read VT is int16_t != exptected VT");
            }
            break;
        case ZarrDatatype::INT8:
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
struct ReadZarr<ContiguousTensor<VT>> {
    static void apply(ContiguousTensor<VT> *&res, const char *filename, const ZarrFileMetaData& zfmd, std::shared_ptr<spdlog::logger> log) {

        if (zfmd.chunks != zfmd.shape) {
            throw std::runtime_error("ReadZarr->ContiguousTensor: Mismatch between chunk and tensor shape. Consider using ReadZarr->ChunkedTensor instead.");
        }

        CheckZarrMetaDataVT<VT>(zfmd.data_type);

        std::vector<size_t> chunks_per_dim = computeChunksPerDim(zfmd.chunks, zfmd.shape);

        uint64_t total_elements = zfmd.shape[0];
        for (size_t i = 1; i < chunks_per_dim.size(); i++) {
            total_elements *= zfmd.shape[i];
        }

        // Fetch all avaiable chunk keys within the respective directory
        std::string base_file_path                                         = filename;
        std::vector<std::pair<std::string, std::string>> chunk_keys_in_dir = GetAllChunkKeys(base_file_path);

        // Check retrieved keys for vailidity and generate full canonical path and ascociated chunk ids from it
        std::vector<std::vector<size_t>> chunk_ids;
        std::vector<std::string> full_chunk_file_paths;
        for (size_t i = 0; i < chunk_keys_in_dir.size(); i++) {
            auto tmp = GetChunkIdsFromChunkKey(std::get<1>(chunk_keys_in_dir[i]), zfmd.dimension_separator, zfmd.shape, chunks_per_dim);

            if (tmp) {
                full_chunk_file_paths.push_back(
                  std::filesystem::canonical(std::filesystem::path(std::get<0>(chunk_keys_in_dir[i]))));
                chunk_ids.push_back(tmp.value());
            }
        }

        if (!full_chunk_file_paths.empty()) {
            throw std::runtime_error("ReadZarr->ContiguousTensor: Found more than one chunk");
        }

        bool endianness_match = checkEndiannessMatch(zfmd.byte_order, log);

        res = DataObjectFactory::create<ContiguousTensor<VT>>(zfmd.shape, InitCode::NONE);

        // For all requested chunks open the respective file, fetch the ptr to the chunks location in the chunked tensor
        // and directly read into it
        for (size_t i = 0; i < full_chunk_file_paths.size(); i++) {
            // IO via STL -> Posix ; substitue io_uring calls here
            std::ifstream f;
            f.open(full_chunk_file_paths[i], std::ios::in | std::ios::binary);

            if (!f.good()) {
                throw std::runtime_error("ReadZarr->ContiguousTensor: failed to open chunk file.");
            }

            uint64_t amount_of_bytes_to_read = sizeof(VT) * total_elements;
            f.read(reinterpret_cast<char *>(res->data.get()), amount_of_bytes_to_read);

            if (!f.good()) {
                throw std::runtime_error("ReadZarr->ContiguousTensor: failed to read chunk file.");
            }

            // Files endianness does not match the native endianness -> byte reverse every read element in the read
            // chunk
            if (!endianness_match) {
                ReverseArray(res->data.get(), total_elements);
            }
        }
    }
};

template<typename VT>
struct ReadZarr<ChunkedTensor<VT>> {
    static void apply(ChunkedTensor<VT> *&res, const char *filename, const ZarrFileMetaData& zfmd, std::shared_ptr<spdlog::logger> log) {

        if (zfmd.shape.size() != zfmd.chunks.size()) {
            log->error("ReadZarr->ChunkedTensor:  Dimension of tensor shape and chunk shape are mismatched");
            throw std::runtime_error("ReadZarr->ChunkedTensor: Dimension of tensor shape and chunk shape are missmatched");
        }

        CheckZarrMetaDataVT<VT>(zfmd.data_type);

        std::vector<size_t> chunks_per_dim = computeChunksPerDim(zfmd.chunks, zfmd.shape);
        uint64_t elements_per_chunk = computeElementsPerChunk(zfmd.chunks, chunks_per_dim.size());

        // Fetch all avaiable chunk keys within the respective directory
        std::string base_file_path                                         = filename;
        std::vector<std::pair<std::string, std::string>> chunk_keys_in_dir = GetAllChunkKeys(base_file_path);

        // Check retrieved keys for vailidity and generate full canonical path and ascociated chunk ids from it
        std::vector<std::vector<size_t>> chunk_ids;
        std::vector<std::string> full_chunk_file_paths;
        for (size_t i = 0; i < chunk_keys_in_dir.size(); i++) {
            auto tmp = GetChunkIdsFromChunkKey(std::get<1>(chunk_keys_in_dir[i]), zfmd.dimension_separator, zfmd.shape, chunks_per_dim);

            if (tmp) {
                full_chunk_file_paths.push_back(std::filesystem::canonical(std::filesystem::path(std::get<0>(chunk_keys_in_dir[i]))));
                chunk_ids.push_back(tmp.value());
            }
        }

        log->info("Reading {} out of {} chunks", chunk_ids.size(), chunk_ids.size());

        bool endianness_match = checkEndiannessMatch(zfmd.byte_order, log);

        res = DataObjectFactory::create<ChunkedTensor<VT>>(zfmd.shape, zfmd.chunks, InitCode::NONE);

        // For all requested chunks open the respective file, fetch the ptr to the chunks location in the chunked
        // tensor and directly read into it
        for (size_t i = 0; i < full_chunk_file_paths.size(); i++) {
            // IO via STL -> Posix ; substitude io_uring calls here
            std::ifstream f;
            f.open(full_chunk_file_paths[i], std::ios::in | std::ios::binary);

            if (!f.good()) {
                log->error("ReadZarr->ChunkedTensor: failed to open chunk file '{}' ", full_chunk_file_paths[i]);
                throw std::runtime_error("ReadZarr->ChunkedTensor: failed to open chunk file.");
            }

            uint64_t amount_of_bytes_to_read = sizeof(VT) * elements_per_chunk;
            f.read(reinterpret_cast<char *>(res->getPtrToChunk(chunk_ids[i])), amount_of_bytes_to_read);

            if (!f.good()) {
                log->error("ReadZarr->ChunkedTensor: failed to read chunk file '{}'", full_chunk_file_paths[i]);
                throw std::runtime_error("ReadZarr->ChunkedTensor: failed to read chunk file.");
            }

            // Files endianness does not match the native endianness -> byte reverse every read element in the read chunk
            if (!endianness_match) {
                ReverseArray(res->data.get(), elements_per_chunk);
            }

            res->chunk_materialization_flags[res->getLinearChunkIdFromChunkIds(chunk_ids[i])] = true;
        }
    }
};

// As in the tensor classes themselves the ranges are inclusive on both sides

template<typename VT>
struct PartialReadZarr<ContiguousTensor<VT>> {
    static void apply(ContiguousTensor<VT> *&res,
                      const char *filename,
                      const ZarrFileMetaData& zfmd,
                      const std::vector<std::pair<size_t, size_t>> &element_id_ranges,
                      std::shared_ptr<spdlog::logger> log) {}
};

template<typename VT>
struct PartialReadZarr<ChunkedTensor<VT>> {
    static void apply(ChunkedTensor<VT> *&res, const char *filename, const ZarrFileMetaData& zfmd, const std::vector<std::pair<size_t, size_t>> &element_id_ranges, std::shared_ptr<spdlog::logger> log) {

        if (zfmd.shape.size() != zfmd.chunks.size()) {
            throw std::runtime_error("PartialReadZarr->ChunkedTensor: Dimension of tensor shape and chunk shape are missmatched");
        }

        CheckZarrMetaDataVT<VT>(zfmd.data_type);

        std::vector<size_t> chunks_per_dim;
        chunks_per_dim.resize(zfmd.shape.size());
        uint64_t total_number_of_chunks = 1;
        for (size_t i = 0; i < chunks_per_dim.size(); i++) {
            chunks_per_dim[i] = zfmd.shape[i] / zfmd.chunks[i];
            total_number_of_chunks *= chunks_per_dim[i];
        }
        uint64_t elements_per_chunk = computeElementsPerChunk(zfmd.chunks, chunks_per_dim.size());

        // Fetch all avaiable chunk keys within the respective directory
        std::string base_file_path                                         = filename;
        std::vector<std::pair<std::string, std::string>> chunk_keys_in_dir = GetAllChunkKeys(base_file_path);

        // Check retrieved keys for vailidity and generate full canonical path and ascociated chunk ids from it
        std::vector<std::vector<size_t>> chunk_ids;
        std::vector<std::string> full_chunk_file_paths;
        for (size_t i = 0; i < chunk_keys_in_dir.size(); i++) {
            auto tmp = GetChunkIdsFromChunkKey(std::get<1>(chunk_keys_in_dir[i]), zfmd.dimension_separator, zfmd.shape, chunks_per_dim);

            if (tmp) {
                full_chunk_file_paths.push_back(std::filesystem::canonical(std::filesystem::path(std::get<0>(chunk_keys_in_dir[i]))));
                chunk_ids.push_back(tmp.value());
            }
        }

        res = DataObjectFactory::create<ChunkedTensor<VT>>(zfmd.shape, zfmd.chunks, InitCode::NONE);

        // Convert element ranges into list of chunks required
        std::optional<std::vector<std::vector<size_t>>> requested_chunk_ids = res->GetChunkListFromIdRange(element_id_ranges);

        if (!requested_chunk_ids) {
            throw std::runtime_error("PartialReadZarr->ChunkedTensor: Invalid element range. Range out-of-bounds or has mismatching dimension");
        }

        std::stringstream ss;
        for (const auto& e : requested_chunk_ids.value()) {
            ss << "[";
            for (const auto& e1 : e) {
                ss << e1 << ", ";
            }
            ss << "]";
        }

        log->debug("Reading {} out of {} chunks", requested_chunk_ids.value().size(), total_number_of_chunks);

        bool endianness_match = checkEndiannessMatch(zfmd.byte_order, log);

        // Match requested chunks to the available chunks in the fs, discard not-requested files and throw on missing file
        std::vector<std::string> full_requested_chunk_file_paths = computeFullFilePathsForRequestedChunks(requested_chunk_ids.value(), full_chunk_file_paths, chunk_ids);

        // For all requested chunks open the respective file, fetch the ptr to the chunks location in the chunked
        // tensor and directly read into it
        for (size_t i = 0; i < full_requested_chunk_file_paths.size(); i++) {
            // IO via STL -> Posix ; substitude io_uring calls here
            std::ifstream f;
            f.open(full_requested_chunk_file_paths[i], std::ios::in | std::ios::binary);

            if (!f.good()) {
                throw std::runtime_error("PartialReadZarr->ChunkedTensor: failed to open chunk file.");
            }

            uint64_t amount_of_bytes_to_read = sizeof(VT) * elements_per_chunk;
            f.read(reinterpret_cast<char *>(res->getPtrToChunk(requested_chunk_ids.value()[i])), amount_of_bytes_to_read);

            if (!f.good()) {
                throw std::runtime_error("PartialReadZarr->ChunkedTensor: failed to read chunk file.");
            }

            // Files endianness does not match the native endianness -> byte reverse every read element in the read chunk
            if (!endianness_match) {
                ReverseArray(res->data.get(), elements_per_chunk);
            }

            res->chunk_materialization_flags[res->getLinearChunkIdFromChunkIds(requested_chunk_ids.value()[i])] = true;
        }
    }
};

#endif    // ZARR_IO_H
