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

#include <parser/metadata/ZarrFileMetaDataParser.h>

#include <fstream>
#include <string>

#include <runtime/local/io/ZarrFileMetaData.h>
#include <nlohmannjson/json.hpp>
#include <stdexcept>

using nlohmann::json;

const std::string ZarrFileMetaDataParser::ZARR_KEY_FILE_EXTENSION = "/.zarray";
const std::string ZarrFileMetaDataParser::ZARR_KEY_CHUNKS = "chunks";
const std::string ZarrFileMetaDataParser::ZARR_KEY_SHAPE = "shape";
const std::string ZarrFileMetaDataParser::ZARR_KEY_FORMAT = "zarr_format";
const std::string ZarrFileMetaDataParser::ZARR_KEY_ORDER = "order";
const std::string ZarrFileMetaDataParser::ZARR_KEY_FILLVALUE = "fill_value";
const std::string ZarrFileMetaDataParser::ZARR_KEY_DTYPE = "dtype";
const std::string ZarrFileMetaDataParser::ZARR_KEY_DIMENSION_SEPARATOR = "dimension_separator";
const std::string ZarrFileMetaDataParser::ZARR_KEY_COMPRESSOR = "compressor";
const std::string ZarrFileMetaDataParser::ZARR_KEY_FILTERS = "filters";

const char ZarrFileMetaDataParser::ZARR_KEY_VAL_BYTEORDER_LE = '<';
const char ZarrFileMetaDataParser::ZARR_KEY_VAL_BYTEORDER_BE = '>';
const char ZarrFileMetaDataParser::ZARR_KEY_VAL_BYTEORDER_NOT_RELEVANT = '|';

const char ZarrFileMetaDataParser::ZARR_KEY_VAL_BYTELENGTH_1 = '1';
const char ZarrFileMetaDataParser::ZARR_KEY_VAL_BYTELENGTH_2 = '2';
const char ZarrFileMetaDataParser::ZARR_KEY_VAL_BYTELENGTH_4 = '4';
const char ZarrFileMetaDataParser::ZARR_KEY_VAL_BYTELENGTH_8 = '8';


const char ZarrFileMetaDataParser::ZARR_KEY_VAL_DTYPE_INTEGER = 'i';
const char ZarrFileMetaDataParser::ZARR_KEY_VAL_DTYPE_BOOLEAN = 'b';
const char ZarrFileMetaDataParser::ZARR_KEY_VAL_DTYPE_UNSIGNED_INTEGER = 'u';
const char ZarrFileMetaDataParser::ZARR_KEY_VAL_DTYPE_FLOAT = 'f';
const char ZarrFileMetaDataParser::ZARR_KEY_VAL_DTYPE_COMPLEX_FLOATING = 'c';
const char ZarrFileMetaDataParser::ZARR_KEY_VAL_DTYPE_TIMEDELTA = 'm';
const char ZarrFileMetaDataParser::ZARR_KEY_VAL_DTYPE_DATETIME = 'M';
const char ZarrFileMetaDataParser::ZARR_KEY_VAL_DTYPE_STRING = 'S';
const char ZarrFileMetaDataParser::ZARR_KEY_VAL_DTYPE_UNICODE = 'U';
const char ZarrFileMetaDataParser::ZARR_KEY_VAL_DTYPE_OTHER = 'V';

ZarrFileMetaData ZarrFileMetaDataParser::readMetaData(const std::string& filename) {
    std::ifstream zarr_meta_file(filename + ZARR_KEY_FILE_EXTENSION);

    if (!zarr_meta_file.good() || !zarr_meta_file.is_open()) {
        throw std::runtime_error("Error while opening Zarr meta data file `" + filename + "`");
    }
    auto zarr_metadata = json::parse(zarr_meta_file);

    ZarrFileMetaData zfmd { /*.chunks=*/ zarr_metadata[ZARR_KEY_CHUNKS].get<decltype(zfmd.chunks)>(),
                            /*.shape=*/ zarr_metadata[ZARR_KEY_SHAPE].get<decltype(zfmd.shape)>(),
                            /*.zarr_format=*/ zarr_metadata[ZARR_KEY_FORMAT].get<decltype(zfmd.zarr_format)>(),
                            /*.order=*/ zarr_metadata[ZARR_KEY_ORDER].get<decltype(zfmd.order)>(),
                            /*.fill_value=*/ zarr_metadata[ZARR_KEY_FILLVALUE].get<decltype(zfmd.fill_value)>(),
                            /*.dtype=*/ zarr_metadata[ZARR_KEY_DTYPE].get<decltype(zfmd.dtype)>()};

    // extract byte order
    switch (zfmd.dtype.at(0)) {
        case ZARR_KEY_VAL_BYTEORDER_LE: zfmd.byte_order = ByteOrder::LITTLEENDIAN; break;
        case ZARR_KEY_VAL_BYTEORDER_BE: zfmd.byte_order = ByteOrder::BIGENDIAN; break;
        case ZARR_KEY_VAL_BYTEORDER_NOT_RELEVANT: zfmd.byte_order = ByteOrder::NOT_RELEVANT; break;
        default: break;
    }

    // extract data type
    switch (zfmd.dtype.at(1)) {
        using enum ZarrDatatype;

        case ZARR_KEY_VAL_DTYPE_BOOLEAN:
            zfmd.data_type = BOOLEAN;
            break;
        case ZARR_KEY_VAL_DTYPE_INTEGER:
            switch (zfmd.dtype.at(2)) {
                case ZARR_KEY_VAL_BYTELENGTH_8:
                    zfmd.data_type = INT64;
                    break;
                case ZARR_KEY_VAL_BYTELENGTH_4:
                    zfmd.data_type = INT32;
                    break;
                case ZARR_KEY_VAL_BYTELENGTH_2:
                    zfmd.data_type = INT16;
                    break;
                case ZARR_KEY_VAL_BYTELENGTH_1:
                    zfmd.data_type = INT8;
                    break;
                default:
                    throw std::runtime_error("Zarr meta data file parsing: Unsupported bit width of value type encountered.");
            }
            break;
        case ZARR_KEY_VAL_DTYPE_UNSIGNED_INTEGER:
            switch (zfmd.dtype.at(2)) {
                case ZARR_KEY_VAL_BYTELENGTH_8:
                    zfmd.data_type = UINT64;
                    break;
                case ZARR_KEY_VAL_BYTELENGTH_4:
                    zfmd.data_type = UINT32;
                    break;
                case ZARR_KEY_VAL_BYTELENGTH_2:
                    zfmd.data_type = UINT16;
                    break;
                case ZARR_KEY_VAL_BYTELENGTH_1:
                    zfmd.data_type = UINT8;
                    break;
                default:
                    throw std::runtime_error("Zarr meta data file parsing: Unsupported bit width of value type encountered.");
            }
            break;
        case ZARR_KEY_VAL_DTYPE_FLOAT:
            switch (zfmd.dtype.at(2)) {
                case ZARR_KEY_VAL_BYTELENGTH_8:
                    zfmd.data_type = FP64;
                    break;
                case ZARR_KEY_VAL_BYTELENGTH_4:
                    zfmd.data_type = FP32;
                    break;
                default:
                    throw std::runtime_error("Zarr meta data file parsing: Unsupported bit width of value type encountered.");
            }
            break;
        case ZARR_KEY_VAL_DTYPE_COMPLEX_FLOATING:
            throw std::runtime_error("Zarr implementation currently does not support complex floats");
        case ZARR_KEY_VAL_DTYPE_TIMEDELTA:
            throw std::runtime_error("Zarr implementation currently does not support time deltas");
        case ZARR_KEY_VAL_DTYPE_DATETIME:
            throw std::runtime_error("Zarr implementation currently does not support dates");
        case ZARR_KEY_VAL_DTYPE_STRING:
            throw std::runtime_error("Zarr implementation currently does not support strings");
        case ZARR_KEY_VAL_DTYPE_UNICODE:
            throw std::runtime_error("Zarr implementation currently does not support unicode");
        case ZARR_KEY_VAL_DTYPE_OTHER:
            throw std::runtime_error("Zarr implementation currently does not support binary blobs");
        default:
            throw std::runtime_error("Zarr meta data file parsing: Unsupported value type encountered.");
    }

    // extract type width in bytes
    zfmd.nBytes = std::stoul(zfmd.dtype.substr(2, zfmd.dtype.size() - 1));

    if (zfmd.nBytes > 1 && zfmd.byte_order == ByteOrder::NOT_RELEVANT) {
        throw std::runtime_error("Zarr metadata specifies a datatype with size > 1B, but specifies that the ByteOrder is irrelevant");
    }

    if (zarr_metadata.contains(ZARR_KEY_DIMENSION_SEPARATOR)) {
        zfmd.dimension_separator = zarr_metadata[ZARR_KEY_DIMENSION_SEPARATOR].get<decltype(zfmd.dimension_separator)>();
    }

    if (zarr_metadata.contains(ZARR_KEY_COMPRESSOR)) {
        if (!zarr_metadata[ZARR_KEY_COMPRESSOR].is_null()) {
            throw std::runtime_error("Daphne Zarr implementation does not support compression yet");
        }
    }

    if (zarr_metadata.contains(ZARR_KEY_FILTERS)) {
        if (!zarr_metadata[ZARR_KEY_FILTERS].is_null()) {
            throw std::runtime_error("Daphne Zarr implementation does not support filters yet");
        }
    }

    if (zfmd.shape.empty() || zfmd.chunks.empty()) {
        throw std::runtime_error("Tensors with rank/dimensionalty 0, i.e., scalars, are not supported");
    }

    if (zfmd.chunks.size() != zfmd.shape.size()) {
        throw std::runtime_error("Rank of tensor does not match that of the specified chunk shape (i.e., chunk_shape.size() != tensor_shape.size())");
    }

    return zfmd;
}
