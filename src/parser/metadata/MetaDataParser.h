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

#pragma once

#include <nlohmannjson/json.hpp>

#include <runtime/local/io/FileMetaData.h>
#include <runtime/local/datastructures/ValueTypeCode.h>

#include <string>

// must be in the same namespace as the enum class ValueTypeCode
NLOHMANN_JSON_SERIALIZE_ENUM(ValueTypeCode, {
    { ValueTypeCode::INVALID, nullptr },
    { ValueTypeCode::SI8, "si8" },
    { ValueTypeCode::SI32, "si32" },
    { ValueTypeCode::SI64, "si64" },
    { ValueTypeCode::UI8, "ui8" },
    { ValueTypeCode::UI32, "ui32" },
    { ValueTypeCode::UI64, "ui64" },
    { ValueTypeCode::F32, "f32" },
    { ValueTypeCode::F64, "f64" }
})

/**
 * @brief A JSON representation of a schema column needed to serialize/deserialize
 * it to/from JSON format.
 */
class SchemaColumn {
public:
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(SchemaColumn, label, valueType)
    [[nodiscard]] const std::string& getLabel() const { return label; }
    [[nodiscard]] ValueTypeCode getValueType() const { return valueType; }

private:
    std::string label;
    ValueTypeCode valueType;
};

class MetaDataParser {

public:
    /**
     * @brief Retrieves the file meta data for the specified file.
     *
     * @param filename The name of the file for which to retrieve the metadata.
     * Metadata should be passed using a simple JSON-based format.
     * @return The meta data of the specified file.
     * @throws std::runtime_error Thrown if the specified file could not be open.
     * @throws std::invalid_argument Thrown if the JSON file contains any unexpected
     * keys or if the file doesn't contain all the metadata.
     */
    static FileMetaData readMetaData(const std::string& filename);

private:
    /**
     * @brief Checks whether a specified key exists in JSON or not.
     *
     * @param j An object that stores JSON.
     * @param key A JSON key.
     * @return True if the key exists; otherwise, false.
     */
    static bool keyExists(const nlohmann::json& j, const std::string& key);
};
