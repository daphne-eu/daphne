#include <util/PropertyLogger.h>
#include <fstream>

PropertyLogger& PropertyLogger::instance() {
    static PropertyLogger instance;
    return instance;
}

void PropertyLogger::logProperty(uint32_t value_id, PropertyValue value) {
    std::lock_guard<std::mutex> lock(mutex);
    properties[value_id].push_back(std::move(value));
}

void PropertyLogger::savePropertiesAsJson(const std::string& filename) const {
    nlohmann::json j;
    {
        std::lock_guard<std::mutex> lock(mutex);

        if (properties.empty()) {
        }

        for (const auto& opEntry : properties) {
            const uint32_t value_id = opEntry.first;
            const auto& propVec = opEntry.second;

            nlohmann::json propJson;
            for (const auto& prop : propVec) {
                prop->to_json(propJson);
            }
            j[std::to_string(value_id)] = propJson;
        }
    }

    std::ofstream file(filename);
    if (file.is_open()) {
        file << j.dump(4);
    } else {
        throw std::runtime_error("Failed to open file: " + filename);
    }
}

std::vector<const Property*> PropertyLogger::getProperties(uint32_t value_id) const {
    std::lock_guard<std::mutex> lock(mutex);
    std::vector<const Property*> result;
    auto it = properties.find(value_id);
    if (it != properties.end()) {
        for (const auto& prop : it->second) {
            result.push_back(prop.get());
        }
    }
    return result;
}