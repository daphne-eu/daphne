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

#include <endian.h>
#include <parser/metadata/ZarrFileMetaDataParser.h>

#include <fstream>
#include <iostream>

#include <runtime/local/io/ZarrFileMetadata.h>
#include <nlohmannjson/json.hpp>

using nlohmann::json;

ZarrFileMetaData ZarrFileMetaDataParser::readMetaData(const std::string& filename) {
    std::ifstream f(filename + "/.zarray");
    // todo: check if file stream is GOOD()
    auto data = json::parse(f);

    ZarrFileMetaData zfmd { /*.chunks=*/ data["chunks"].get<decltype(zfmd.chunks)>(),
                            /*.shape=*/ data["shape"].get<decltype(zfmd.shape)>(),
                            /*.zarr_format=*/ data["zarr_format"].get<decltype(zfmd.zarr_format)>(),
                            /*.order=*/ data["order"].get<decltype(zfmd.order)>(),
                            /*.fill_value=*/ data["fill_value"].get<decltype(zfmd.fill_value)>(),
                            /*.dtype=*/ data["dtype"].get<decltype(zfmd.dtype)>(),
                            // Values will be overwriten later. Init to apease compiler
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
        case 'b':
            zfmd.data_type = ZarrDatatype::BOOLEAN;
            break;
        case 'i':
            switch (zfmd.dtype.at(2)) {
                case '8':
                    zfmd.data_type = ZarrDatatype::INT64;
                    break;
                case '4':
                    zfmd.data_type = ZarrDatatype::INT32;
                    break;
                case '2':
                    zfmd.data_type = ZarrDatatype::INT16;
                    break;
                case '1':
                    zfmd.data_type = ZarrDatatype::INT8;
                    break;
                default:
                    throw std::runtime_error("Zarr meta data file parsing: Unsupported bit width of VT encountred.");
            }
            break;
        case 'u':
            switch (zfmd.dtype.at(2)) {
                case '8':
                    zfmd.data_type = ZarrDatatype::UINT64;
                    break;
                case '4':
                    zfmd.data_type = ZarrDatatype::UINT32;
                    break;
                case '2':
                    zfmd.data_type = ZarrDatatype::UINT16;
                    break;
                case '1':
                    zfmd.data_type = ZarrDatatype::UINT8;
                    break;
                default:
                    throw std::runtime_error("Zarr meta data file parsing: Unsupported bit width of VT encountred.");
            }
            break;
        case 'f':
            switch (zfmd.dtype.at(2)) {
                case '8':
                    zfmd.data_type = ZarrDatatype::FP64;
                    break;
                case '4':
                    zfmd.data_type = ZarrDatatype::FP32;
                    break;
                default:
                    throw std::runtime_error("Zarr meta data file parsing: Unsupported bit width of VT encountred.");
            }
            break;
        case 'c':
            zfmd.data_type = ZarrDatatype::COMPLEX_FLOATING;
            break;
        case 'm':
            zfmd.data_type = ZarrDatatype::TIMEDELTA;
            break;
        case 'M':
            zfmd.data_type = ZarrDatatype::DATETIME;
            break;
        case 'S':
            zfmd.data_type = ZarrDatatype::STRING;
            break;
        case 'U':
            zfmd.data_type = ZarrDatatype::UNICODE;
            break;
        case 'V':
            zfmd.data_type = ZarrDatatype::OTHER;
            break;
        default:
            throw std::runtime_error("Zarr meta data file parsing: Unsupported VT encountered.");
            break;
    }

    // extract type width in bytes
    zfmd.nBytes = std::stoul(zfmd.dtype.substr(2, zfmd.dtype.size() - 1));

    if (data.contains("dimension_separator")) {
        zfmd.dimension_separator = data["dimension_separator"].get<decltype(zfmd.dimension_separator)>();
    }

    if (data.contains("compressor")) {
        //std::cout << "Compression not yet supported!" << std::endl;
    }

    if (data.contains("filters")) {
        //std::cout << "Filters not yet supported!" << std::endl;
    }

    return zfmd;
}
