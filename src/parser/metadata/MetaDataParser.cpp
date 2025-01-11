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
#include <parser/metadata/MetaDataParser.h>
#include <parser/metadata/JsonKeys.h>
#include <runtime/local/kernels/Read.h>
#include <runtime/local/io/utils.h>

#include <filesystem>
#include <fstream>
#include <iostream>

FileMetaData MetaDataParser::readMetaData(const std::string &filename_, bool labels) {
    std::string metaFilename = filename_ + ".meta";
    std::ifstream ifs(metaFilename, std::ios::in);
    if (!ifs.good()){
        int extv = extValue(&filename_[0]);
        //TODO: Support other file types than csv
        if (extv == 0) {
            FileMetaData fmd = generateFileMetaData(filename_, labels);
            try {
                writeMetaData(filename_, fmd);
            } catch (std::exception &e) {
                std::cerr << "Could not write generated meta data to file '" << metaFilename << "': " << e.what()
                          << std::endl;
            }
            return fmd;
        }
        throw std::runtime_error("Could not open file '" + metaFilename + "' for reading meta data. \n"
        + "Note: meta data file generation is currently only supported for csv files");
    }

    std::stringstream buffer;
    buffer << ifs.rdbuf();
    return MetaDataParser::readMetaDataFromString(buffer.str());
}
FileMetaData MetaDataParser::readMetaDataFromString(const std::string &str) {
    nlohmann::json jf = nlohmann::json::parse(str);

    if (!keyExists(jf, JsonKeys::NUM_ROWS) || !keyExists(jf, JsonKeys::NUM_COLS)) {
        throw std::invalid_argument("A meta data JSON file should always contain \"" + JsonKeys::NUM_ROWS +
                                    "\" and \"" + JsonKeys::NUM_COLS + "\" keys.");
    }

    const size_t numRows = jf.at(JsonKeys::NUM_ROWS).get<size_t>();
    const size_t numCols = jf.at(JsonKeys::NUM_COLS).get<size_t>();
    const bool isHDFS = (keyExists(jf, JsonKeys::HDFS));
    const bool isSingleValueType = !(keyExists(jf, JsonKeys::SCHEMA));
    const ssize_t numNonZeros =
        (keyExists(jf, JsonKeys::NUM_NON_ZEROS)) ? jf.at(JsonKeys::NUM_NON_ZEROS).get<ssize_t>() : -1;

    HDFSMetaData hdfs;
    if (isHDFS) {
        // TODO check if key exist and throw errors if not
        hdfs.isHDFS = jf.at(JsonKeys::HDFS)["isHDFS"];
        ;
        hdfs.HDFSFilename = jf.at(JsonKeys::HDFS)["HDFSFilename"];
    }
    if (isSingleValueType) {
        if (keyExists(jf, JsonKeys::VALUE_TYPE)) {
            ValueTypeCode vtc = jf.at(JsonKeys::VALUE_TYPE).get<ValueTypeCode>();
            return {numRows, numCols, isSingleValueType, vtc, numNonZeros, hdfs};
        } else {
            throw std::invalid_argument("A (matrix) meta data JSON file should contain the \"" + JsonKeys::VALUE_TYPE +
                                        "\" key.");
        }
    } else {
        if (keyExists(jf, JsonKeys::SCHEMA)) {
            ValueTypeCode default_vtc = ValueTypeCode::INVALID;
            if (keyExists(jf, JsonKeys::VALUE_TYPE)) {
                default_vtc = jf.at(JsonKeys::VALUE_TYPE).get<ValueTypeCode>();
            }
            std::vector<ValueTypeCode> schema;
            std::vector<std::string> labels;
            auto schemaColumn = jf.at(JsonKeys::SCHEMA).get<std::vector<SchemaColumn>>();
            for (const auto &column : schemaColumn) {
                auto vtc = column.getValueType();
                if (vtc == ValueTypeCode::INVALID) {
                    vtc = default_vtc;
                    if (default_vtc == ValueTypeCode::INVALID)
                        throw std::invalid_argument("While reading a frame's meta data, a column "
                                                    "without value type was "
                                                    "found while not providing a default value type.");
                }
                schema.emplace_back(vtc);
                labels.emplace_back(column.getLabel());
            }
            return {numRows, numCols, isSingleValueType, schema, labels, numNonZeros, hdfs};
        } else {
            throw std::invalid_argument("A (frame) meta data JSON file should contain the \"" + JsonKeys::SCHEMA +
                                        "\" key.");
        }
    }
}

std::string MetaDataParser::writeMetaDataToString(const FileMetaData &metaData) {
    nlohmann::json json;

    json[JsonKeys::NUM_ROWS] = metaData.numRows;
    json[JsonKeys::NUM_COLS] = metaData.numCols;

    if (metaData.isSingleValueType) {
        if (metaData.schema.size() != 1)
            throw std::runtime_error("inappropriate meta data tried to be written to file");
        json[JsonKeys::VALUE_TYPE] = metaData.schema[0];
    } else {
        std::vector<SchemaColumn> schemaColumns;
        // assume that the schema and labels are the same lengths
        for (unsigned int i = 0; i < metaData.schema.size(); i++) {
            SchemaColumn schemaColumn;
            schemaColumn.setLabel(metaData.labels[i]);
            schemaColumn.setValueType(metaData.schema[i]);
            schemaColumns.emplace_back(schemaColumn);
        }
        json[JsonKeys::SCHEMA] = schemaColumns;
    }

    if (metaData.numNonZeros != -1)
        json[JsonKeys::NUM_NON_ZEROS] = metaData.numNonZeros;

    // HDFS
    if (metaData.hdfs.isHDFS) {
        json[JsonKeys::HDFS][JsonKeys::HDFSKeys::isHDFS] = metaData.hdfs.isHDFS;
        std::filesystem::path filePath(metaData.hdfs.HDFSFilename);
        auto baseFileName = filePath.filename().string();

        json[JsonKeys::HDFS][JsonKeys::HDFSKeys::HDFSFilename] = "/" + baseFileName;
    }
    return json.dump();
}
void MetaDataParser::writeMetaData(const std::string &filename_, const FileMetaData &metaData) {
    std::string metaFilename = filename_ + ".meta";
    std::ofstream ofs(metaFilename, std::ios::out);
    if (!ofs.good())
        throw std::runtime_error("could not open file '" + metaFilename + "' for writing meta data");

    if (ofs.is_open()) {
        ofs << MetaDataParser::writeMetaDataToString(metaData);
    } else
        throw std::runtime_error("could not open file '" + metaFilename + "' for writing meta data");
}

bool MetaDataParser::keyExists(const nlohmann::json &j, const std::string &key) { return j.find(key) != j.end(); }
