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

#include <fstream>

FileMetaData MetaDataParser::readMetaData(const std::string& filename) {
    std::ifstream ifs(filename, std::ios::in);
    if (!ifs.good())
        throw std::runtime_error("Could not open file '" + filename + "' for reading meta data.");

    nlohmann::basic_json jf = nlohmann::json::parse(ifs);

    if (!keyExists(jf, JsonKeys::NUM_ROWS) || !keyExists(jf, JsonKeys::NUM_COLS))
        throw std::invalid_argument("A meta data JSON file should always contain \"" + JsonKeys::NUM_ROWS + "\" and \"" + JsonKeys::NUM_COLS + "\" keys.");

    const size_t numRows = jf.at(JsonKeys::NUM_ROWS).get<size_t>();
    const size_t numCols = jf.at(JsonKeys::NUM_COLS).get<size_t>();

    const bool isSingleValueType = !(keyExists(jf, JsonKeys::SCHEMA));

    std::vector<ValueTypeCode> schema;
    std::vector<std::string> labels;

    if (isSingleValueType) {
        if (keyExists(jf, JsonKeys::VALUE_TYPE)) {
            ValueTypeCode vtc = jf.at(JsonKeys::VALUE_TYPE).get<ValueTypeCode>();
            schema.emplace_back(vtc);
        } else throw std::invalid_argument("A (matrix) meta data JSON file should contain the \"" + JsonKeys::VALUE_TYPE + "\" key.");
    } else {
        if (keyExists(jf, JsonKeys::SCHEMA)) {
            auto schemaColumn = jf.at(JsonKeys::SCHEMA).get<std::vector<SchemaColumn>>();
            for (const auto& row: schemaColumn) {
                schema.emplace_back(row.getValueType());
                labels.emplace_back(row.getLabel());
            }
        } else throw std::invalid_argument("A (frame) meta data JSON file should contain the \"" + JsonKeys::SCHEMA + "\" key.");
    }

    const ssize_t numNonZeros = (keyExists(jf, JsonKeys::NUM_NON_ZEROS)) ? jf.at(JsonKeys::NUM_NON_ZEROS).get<ssize_t>() : -1;

    return FileMetaData(numRows, numCols, isSingleValueType, schema, labels, numNonZeros);
}

bool MetaDataParser::keyExists(const nlohmann::json& j, const std::string& key) { return j.find(key) != j.end(); }