#pragma once

#include <cstddef>
#include <nlohmannjson/json.hpp>
#include <unordered_map>
#include <string>
#include <mutex>
#include <fstream>
#include <variant>

using PropertyValue = std::variant<std::string, size_t, std::pair<size_t,size_t>>;

struct PropertyLogger {
    std::unordered_map<int64_t, std::unordered_map<std::string, PropertyValue>> properties;
    std::mutex mutex;

    PropertyLogger(const PropertyLogger&) = delete;
    PropertyLogger& operator=(const PropertyLogger&) = delete;

    static PropertyLogger& instance() {
        static PropertyLogger instance;
        return instance;
    }

    void logProperty(int64_t opId, const std::string& key, const PropertyValue& value) {
        std::lock_guard<std::mutex> lock(mutex);
        properties[opId][key] = value;
    }

    void savePropertiesAsJson(const std::string& filename) {
        nlohmann::json j;
        {
            std::lock_guard<std::mutex> lock(mutex);
            for (const auto& opEntry : properties) {
                const int64_t opId = opEntry.first;
                const auto& propMap = opEntry.second;

                nlohmann::json propJson;
                for (const auto& propEntry : propMap) {
                    const std::string& key = propEntry.first;
                    const PropertyValue& value = propEntry.second;

                    std::visit([&propJson, &key](const auto& val) {
                        if constexpr (std::is_same_v<decltype(val), std::string>) {
                            propJson[key] = val;
                        } else if constexpr (std::is_same_v<decltype(val), size_t>) {
                            propJson[key] = val;
                        } else if constexpr (std::is_same_v<decltype(val), std::pair<size_t, size_t>>) {
                            propJson[key] = {val.first, val.second};
                        }
                    }, value);
                }
                j[opId] = propJson;
            }
        }
        std::ofstream file(filename);
        if (file.is_open()) {
            file << j.dump(4);
        } else {
            throw std::runtime_error("Failed to open file: " + filename);
        }
    }

private:
    PropertyLogger() = default;
};
