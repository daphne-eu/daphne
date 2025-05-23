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

#include <ir/daphneir/Daphne.h>
#include <ir/daphneir/Passes.h>

#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Pass/Pass.h>
#include <nlohmannjson/json.hpp>

#include <algorithm>
#include <fstream>
#include <string>

#include <cstdint>

namespace mlir {
#define GEN_PASS_DECL_INSERTPROPERTIESPASS
#define GEN_PASS_DEF_INSERTPROPERTIESPASS
#include "ir/daphneir/Passes.h.inc"
} // namespace mlir

using namespace mlir;

nlohmann::json readPropertiesFromFile(const std::string &filename) {
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

    bool operator<(const PropertyEntry &other) const { return valueID < other.valueID; }
};

namespace {
struct InsertPropertiesPass : public impl::InsertPropertiesPassBase<InsertPropertiesPass> {
    InsertPropertiesPass() = default;

  public:
    explicit InsertPropertiesPass(std::string properties_file_path)
        : impl::InsertPropertiesPassBase<InsertPropertiesPass>() {
        this->properties_file_path = properties_file_path;
    };
    void runOnOperation() override;
};
} // end of anonymous namespace

void InsertPropertiesPass::runOnOperation() {
    func::FuncOp func = getOperation();
    OpBuilder builder(func.getContext());

    nlohmann::json propertiesJson = readPropertiesFromFile(this->properties_file_path);
    llvm::DenseMap<std::pair<Value, Operation *>, Value> castOpMap;

    std::vector<PropertyEntry> properties;
    for (auto it = propertiesJson.begin(); it != propertiesJson.end(); ++it) {
        uint32_t valueID = std::stoi(it.key());
        properties.push_back({valueID, it.value()});
    }
    std::sort(properties.begin(), properties.end());
    size_t propertyIndex = 0;

    auto insertRecordedProperties = [&](Operation *op) {
        size_t numResults = op->getNumResults();
        for (unsigned i = 0; i < numResults && propertyIndex < properties.size(); ++i) {
            Value res = op->getResult(i);
            nlohmann::json &prop = properties[propertyIndex].properties;
            auto it = prop.begin();
            while (it != prop.end()) {
                const std::string &key = it.key();
                const nlohmann::json &value = it.value();
                if (key == "sparsity") {
                    if (value.is_null()) {
                        llvm::errs() << "Error: 'sparsity' is null for property index " << propertyIndex << "\n";
                        ++it;
                        continue;
                    } else if (!value.is_number()) {
                        llvm::errs() << "Error: 'sparsity' is not a number for property index " << propertyIndex
                                     << "\n";
                        ++it;
                        continue;
                    }

                    if (res.getType().isa<daphne::MatrixType>()) {
                        auto mt = res.getType().dyn_cast<daphne::MatrixType>();
                        double sparsity = value.get<double>();
                        if (mt) {
                            if ((llvm::isa<scf::ForOp>(op) || llvm::isa<scf::WhileOp>(op) ||
                                 llvm::isa<scf::IfOp>(op))) {
                                builder.setInsertionPointAfter(op);
                                builder.create<daphne::CastOp>(op->getLoc(), mt.withSparsity(sparsity), res);
                            }

                            else {
                                for (auto &use : res.getUses()) {
                                    Operation *userOp = use.getOwner();
                                    if (isa<scf::ForOp>(userOp) || isa<scf::IfOp>(userOp) ||
                                        isa<scf::WhileOp>(userOp)) {
                                        auto key = std::make_pair(res, userOp);

                                        Value castOpValue;
                                        if (castOpMap.count(key)) {
                                            castOpValue = castOpMap[key];
                                        } else {
                                            builder.setInsertionPoint(userOp);
                                            castOpValue = builder.create<daphne::CastOp>(op->getLoc(), mt, res);
                                            castOpMap[key] = castOpValue;
                                        }

                                        userOp->setOperand(use.getOperandNumber(), castOpValue);
                                    }
                                }
                            }

                            ++propertyIndex;
                        }
                    }
                }
                ++it;
            }
        }
    };

    func.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
        if (propertyIndex >= properties.size())
            return WalkResult::advance();

        // Skip specific ops that should not be processed
        if (isa<daphne::RecordPropertiesOp>(op) || op->hasAttr("daphne.value_ids"))
            return WalkResult::advance();

        if (auto castOp = dyn_cast<daphne::CastOp>(op)) {
            if (castOp.isRemovePropertyCast()) {
                return WalkResult::advance();
            }
        }

        // Handle loops (scf.for and scf.while) and If blocks as black boxes
        if (isa<scf::ForOp>(op) || isa<scf::WhileOp>(op) || isa<scf::IfOp>(op)) {
            insertRecordedProperties(op);
            return WalkResult::skip();
        }

        if (auto funcOp = llvm::dyn_cast<func::FuncOp>(op)) {
            // Check if this is the @main function or a UDF
            if (funcOp.getName() == "main") {
                return WalkResult::advance();
            } else {
                return WalkResult::skip();
            }
        }

        // Process all other operations that output matrix types
        insertRecordedProperties(op);
        return WalkResult::advance();
    });

    if (propertyIndex < properties.size()) {
        llvm::errs() << "Warning: Not all properties were applied."
                     << "\n";
    }
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::daphne::createInsertPropertiesPass(std::string propertiesFilePath) {
    return std::make_unique<InsertPropertiesPass>(propertiesFilePath);
}