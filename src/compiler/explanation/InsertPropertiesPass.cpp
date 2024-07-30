#include "mlir/IR/Builders.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include <ir/daphneir/Passes.h>
#include <ir/daphneir/Daphne.h>
#include <nlohmannjson/json.hpp>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <string>

using namespace mlir;

nlohmann::json readPropertiesFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    nlohmann::json properties;
    file >> properties;
    return properties;
}

struct PropertyEntry {
    uint32_t valueID;
    nlohmann::json properties;

    bool operator<(const PropertyEntry &other) const {
        return valueID < other.valueID;
    }
};

class InsertPropertiesPass : public PassWrapper<InsertPropertiesPass, OperationPass<func::FuncOp>> {
public:
    InsertPropertiesPass(const std::string &jsonFile) : jsonFile(jsonFile) {}

    StringRef getArgument() const final { return "insert-properties"; }
    StringRef getDescription() const final { return "Insert recorded properties back into operations"; }

    void runOnOperation() override {
        func::FuncOp func = getOperation();
        OpBuilder builder(func.getContext());

        nlohmann::json propertiesJson = readPropertiesFromFile(jsonFile);

        std::vector<PropertyEntry> properties;
        for (auto it = propertiesJson.begin(); it != propertiesJson.end(); ++it) {
            uint32_t valueID = std::stoi(it.key());
            properties.push_back({ valueID, it.value() });
        }
        std::sort(properties.begin(), properties.end());

        size_t propertyIndex = 0;

        func.walk([&](Operation *op) {
            for (Value res : op->getResults()) {
                if (propertyIndex >= properties.size()) return;

                nlohmann::json &prop = properties[propertyIndex].properties;
                bool propertyUsed = false;

                for (auto& [key, value] : prop.items()) {
                    if (key == "sparsity") {
                        if (res.getType().isa<daphne::MatrixType>()) {
                            auto mt = res.getType().dyn_cast<daphne::MatrixType>();
                            if (mt) {
                                res.setType(mt.withSparsity(value.get<double>()));
                                propertyUsed = true;
                            }
                        }
                    }
                }

                if (propertyUsed) {
                    ++propertyIndex;
                }
            }
        });

        if (propertyIndex < properties.size()) {
            std::cerr << "Warning: Not all properties were applied." << std::endl;
        }
    }

private:
    std::string jsonFile;
};

std::unique_ptr<Pass> daphne::createInsertPropertiesPass(const std::string jsonFile) {
    return std::make_unique<InsertPropertiesPass>(jsonFile);
}
