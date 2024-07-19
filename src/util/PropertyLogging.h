#pragma once

#include <nlohmann/json.hpp>
#include <unordered_map>
#include <string>
#include <mutex>
#include <fstream>
#include <variant>

using PropertyValue = std::variant<std::string, size_t, double>;

struct PropertyLogger {
    std::unordered_map<std::string, std::unordered_map<std::string, PropertyValue>> properties;
    std::mutex mutex;

    PropertyLogger(const PropertyLogger&) = delete;
    PropertyLogger& operator=(const PropertyLogger&) = delete;

    static PropertyLogger& getInstance() {
        static PropertyLogger instance;
        return instance;
    }

    void logProperty(const std::string& opId, const std::string& key, const PropertyValue& value) {
        std::lock_guard<std::mutex> lock(mutex);
        properties[opId][key] = value;
    }

    void savePropertiesAsJson(const std::string& filename) {
        nlohmann::json j;
        for (const auto& [opId, propMap] : properties) {
            nlohmann::json propJson;
            for (const auto& [key, value] : propMap) {
                std::visit([&propJson, &key](const auto& val) {
                    propJson[key] = val;
                }, value);
            }
            j[opId] = propJson;
        }
        std::ofstream file(filename);
        file << j.dump(4);
    }

private:
    PropertyLogger() = default;
};
