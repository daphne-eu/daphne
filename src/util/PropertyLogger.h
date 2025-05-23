/*
 * Copyright 2025 The DAPHNE Consortium
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

#include <string>
#include <mutex>
#include <memory>
#include <vector>
#include <unordered_map>

struct Property {
    virtual ~Property() = default;
    virtual void to_json(nlohmann::json& j) const = 0;
};

struct StringProperty : public Property {
    std::string key;
    std::string value;

    StringProperty(const std::string& k, const std::string& val) : key(k), value(val) {}

    void to_json(nlohmann::json& j) const override {
        j[key] = value;
    }
};

struct SizeTProperty : public Property {
    std::string key;
    size_t value;

    SizeTProperty(const std::string& k, size_t val) : key(k), value(val) {}

    void to_json(nlohmann::json& j) const override {
        j[key] = value;
    }
};

struct SparsityProperty : public Property {
    double value;

    SparsityProperty(double val) : value(val) {}

    void to_json(nlohmann::json& j) const override {
        j["sparsity"] = value;
    }
};

struct NNZProperty : public Property {
    std::string key;
    size_t value;

    NNZProperty(size_t val) : value(val) {}

    void to_json(nlohmann::json& j) const override {
        j["nnz"] = value;
    }
};

using PropertyValue = std::unique_ptr<Property>;

class PropertyLogger {
public:
    PropertyLogger(const PropertyLogger&) = delete;
    PropertyLogger& operator=(const PropertyLogger&) = delete;

    static PropertyLogger& instance();

    void logProperty(uint32_t value_id, PropertyValue value);

    void savePropertiesAsJson(const std::string& filename) const;

    std::vector<const Property*> getProperties(uint32_t value_id) const;

private:
    PropertyLogger() = default;

    std::unordered_map<int64_t, std::vector<PropertyValue>> properties;
    mutable std::mutex mutex;
};
