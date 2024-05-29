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

#include <parser/metadata/ZarrFileMetaDataParser.h>

#include <fstream>

#include <runtime/local/io/ZarrFileMetadata.h>
#include <nlohmannjson/json.hpp>
#include <stdexcept>

using nlohmann::json;

ZarrFileMetaData ZarrFileMetaDataParser::readMetaData(const std::string& filename) {
    std::ifstream f(filename + "/.zarray");

    if (!f.good() || !f.is_open()) {
        throw std::runtime_error("Error while opening Zarr meta data file");
    }
    auto data = json::parse(f);

    ZarrFileMetaData zfmd { /*.chunks=*/ data["chunks"].get<decltype(zfmd.chunks)>(),
                            /*.shape=*/ data["shape"].get<decltype(zfmd.shape)>(),
                            /*.zarr_format=*/ data["zarr_format"].get<decltype(zfmd.zarr_format)>(),
                            /*.order=*/ data["order"].get<decltype(zfmd.order)>(),
                            /*.fill_value=*/ data["fill_value"].get<decltype(zfmd.fill_value)>(),
                            /*.dtype=*/ data["dtype"].get<decltype(zfmd.dtype)>(),
                            // Values will be overwritten later. Init to appease compiler
                            /*.dimension_seperator*/ ".",
                            /*.byte_order*/ ByteOrder::LITTLEENDIAN,
                            /*.data_type*/ ZarrDatatype::INT64,
                            /*.nBytes*/ 8};

    // extract byte order
    switch (zfmd.dtype.at(0)) {
        case '<': zfmd.byte_order = ByteOrder::LITTLEENDIAN; break;
        case '>': zfmd.byte_order = ByteOrder::BIGENDIAN; break;
        case '|': zfmd.byte_order = ByteOrder::NOT_RELEVANT; break;
        default: break;
    }

    // extract data type
    switch (zfmd.dtype.at(1)) {
        using enum ZarrDatatype;

        case 'b':
            zfmd.data_type = BOOLEAN;
            break;
        case 'i':
            switch (zfmd.dtype.at(2)) {
                case '8':
                    zfmd.data_type = INT64;
                    break;
                case '4':
                    zfmd.data_type = INT32;
                    break;
                case '2':
                    zfmd.data_type = INT16;
                    break;
                case '1':
                    zfmd.data_type = INT8;
                    break;
                default:
                    throw std::runtime_error("Zarr meta data file parsing: Unsupported bit width of VT encountered.");
            }
            break;
        case 'u':
            switch (zfmd.dtype.at(2)) {
                case '8':
                    zfmd.data_type = UINT64;
                    break;
                case '4':
                    zfmd.data_type = UINT32;
                    break;
                case '2':
                    zfmd.data_type = UINT16;
                    break;
                case '1':
                    zfmd.data_type = UINT8;
                    break;
                default:
                    throw std::runtime_error("Zarr meta data file parsing: Unsupported bit width of VT encountered.");
            }
            break;
        case 'f':
            switch (zfmd.dtype.at(2)) {
                case '8':
                    zfmd.data_type = FP64;
                    break;
                case '4':
                    zfmd.data_type = FP32;
                    break;
                default:
                    throw std::runtime_error("Zarr meta data file parsing: Unsupported bit width of VT encountered.");
            }
            break;
        case 'c':
            zfmd.data_type = COMPLEX_FLOATING;
            throw std::runtime_error("Zarr implementation currently does not support complex floats");
            break;
        case 'm':
            zfmd.data_type = TIMEDELTA;
            throw std::runtime_error("Zarr implementation currently does not support time deltas");
            break;
        case 'M':
            zfmd.data_type = DATETIME;
            throw std::runtime_error("Zarr implementation currently does not support dates");
            break;
        case 'S':
            zfmd.data_type = STRING;
            throw std::runtime_error("Zarr implementation currently does not support strings");
            break;
        case 'U':
            zfmd.data_type = UNICODE;
            throw std::runtime_error("Zarr implementation currently does not support unicode");
            break;
        case 'V':
            zfmd.data_type = OTHER;
            throw std::runtime_error("Zarr implementation currently does not support binary blobs");
            break;
        default:
            throw std::runtime_error("Zarr meta data file parsing: Unsupported VT encountered.");
            break;
    }

    // extract type width in bytes
    zfmd.nBytes = std::stoul(zfmd.dtype.substr(2, zfmd.dtype.size() - 1));

    if (zfmd.nBytes > 1 && zfmd.byte_order == ByteOrder::NOT_RELEVANT) {
        throw std::runtime_error("Zarr metadata specifies a datatype with size > 1B, but specifies that the ByteOreder is erelevant");
    }

    if (data.contains("dimension_separator")) {
        zfmd.dimension_separator = data["dimension_separator"].get<decltype(zfmd.dimension_separator)>();
    }

    if (data.contains("compressor")) {
        if (!data["compressor"].is_null()) {
            throw std::runtime_error("Zarr implementation does not support compression yet");
        }    
    }

    if (data.contains("filters")) {
        if (!data["filters"].is_null()) {
            throw std::runtime_error("Zarr implementation does not support filters yet");
        }
    }

    if (zfmd.shape.size() == 0 || zfmd.chunks.size() == 0) {
        throw std::runtime_error("Tensors with rank/dimensionalty 0 i.e. scalars are not supported");
    }

    if (zfmd.chunks.size() != zfmd.shape.size()) {
        throw std::runtime_error("Rank of tensor does not match that of the specified chunk shape. I.e. chunk_shape.size() != tensor_shape.size()");
    }

    return zfmd;
}
