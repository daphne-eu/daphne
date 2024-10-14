#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <ir/daphneir/Passes.h>
#include <ir/daphneir/Daphne.h>
#include <nlohmannjson/json.hpp>
#include <fstream>
#include <algorithm>
#include <string>
#include <mlir/Dialect/SCF/IR/SCF.h>

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

        auto insertRecordedProperties = [&](Operation *op){
            size_t numResults = op->getNumResults();
            for (unsigned i = 0; i < numResults && propertyIndex < properties.size(); ++i)
            {
                Value res = op->getResult(i);
                nlohmann::json &prop = properties[propertyIndex].properties;
                auto it = prop.begin();
                while (it != prop.end()) {
                    const std::string &key = it.key();
                    const nlohmann::json &value = it.value();
                    if(key == "sparsity")
                    {
                        if (value.is_null()) {
                            llvm::errs() << "Error: 'sparsity' is null for property index " << propertyIndex << "\n";
                            ++it;
                            continue;
                        }
                        else if (!value.is_number()) {
                            llvm::errs() << "Error: 'sparsity' is not a number for property index " << propertyIndex << "\n";
                            ++it;
                            continue;
                        }

                        if (res.getType().isa<daphne::MatrixType>()) 
                        {
                            auto mt = res.getType().dyn_cast<daphne::MatrixType>();
                            double sparsity = value.get<double>();
                            if (mt)
                            {
                                if ((llvm::isa<scf::ForOp>(op) || llvm::isa<scf::WhileOp>(op) || llvm::isa<scf::IfOp>(op))
                                    && sparsity < 1.0)
                                {
                                    // Create the CastOp for the current loop result
                                    builder.setInsertionPointAfter(op);
                                    builder.create<daphne::CastOp>(
                                        op->getLoc(),
                                        mt.withSparsity(value.get<double>()),  // Apply the sparsity value
                                        res
                                    );
                                }
                                else 
                                {
                                    res.setType(mt.withSparsity(value.get<double>()));
                                }
                                ++propertyIndex;
                            }
                        }
                    }
                    ++it;
                }

            }
        
        };


        func.walk<WalkOrder::PreOrder>([&](Operation *op)->WalkResult {

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

            /** 
            // Handle (consecutive) CastOp(s) followed by a loop
            else if (auto castOp = llvm::dyn_cast<daphne::CastOp>(op)) {
                Operation *currentOp = op;
                // Check for consecutive cast ops
                while ((currentOp = currentOp->getNextNode()) && llvm::isa<daphne::CastOp>(currentOp)) {
                    // Continue to next CastOp
                }
                // Now check if the final operation is a loop
                if (currentOp && (llvm::isa<scf::ForOp>(currentOp) || llvm::isa<scf::WhileOp>(currentOp) || llvm::isa<scf::IfOp>(currentOp)) ) {
                    // CastOps followed by a loop
                    propertyIndex++;
                    return WalkResult::advance();  // Skip for now, keeping dense
                } 
                else 
                {
                    insertRecordedProperties(op);
                    return WalkResult::advance();
                }
            }
            */

            // Process all other operations that output matrix types
            insertRecordedProperties(op);
            return WalkResult::advance();
            
        });

        if (propertyIndex < properties.size()) {
            llvm::errs() << "Warning: Not all properties were applied." << "\n";
        }
    }

private:
    std::string jsonFile;
};

std::unique_ptr<Pass> mlir::daphne::createInsertPropertiesPass(const std::string jsonFile) {
    return std::make_unique<InsertPropertiesPass>(jsonFile);
}