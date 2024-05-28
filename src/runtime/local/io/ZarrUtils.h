#pragma once

#include <vector>
#include <cstdint>
#include <cstring>
#include <memory>

#include <spdlog/spdlog.h>

#include <runtime/local/io/ZarrUtils.h>
#include <runtime/local/io/ZarrFileMetadata.h>

bool checkEndiannessMatch(const ByteOrder bo, std::shared_ptr<spdlog::logger> log);
std::vector<size_t> computeChunksPerDim(const std::vector<size_t>& chunks, const std::vector<size_t>& shape);
uint64_t computeElementsPerChunk(const std::vector<size_t>& chunks, const size_t n);
std::vector<std::string> computeFullFilePathsForRequestedChunks(
    const std::vector<std::vector<size_t>>& requested_chunk_ids,
    std::vector<std::string>& full_chunk_file_paths,
    std::vector<std::vector<size_t>>& chunk_ids);
