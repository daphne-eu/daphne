#include <util/ErrorHandler.h>
#include <ir/daphneir/Daphne.h>
#include <ir/daphneir/Passes.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Operation.h>
#include <mlir/Pass/Pass.h>

#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream> // Include for debugging output
#include <stdexcept>
#include <memory>
#include <vector>
#include <utility>

using namespace mlir;
using json = nlohmann::json;

namespace {
    class CalculateSparsityPass : public PassWrapper<CalculateSparsityPass, OperationPass<func::FuncOp>> {
        json sparsityResults;

        std::function<WalkResult(Operation*)> walkOp = [&](Operation * op) {
            if(auto matType = op->getResult(0).getType().dyn_cast<daphne::MatrixType>()) {
                std::cout << "Calculating sparsity for operation: " << op->getName().getStringRef().str() << std::endl;
                calculateSparsity(op, matType);
            }

            return WalkResult::advance();
        };

        void calculateSparsity(Operation *op, daphne::MatrixType matType) {
            auto loc = op->getLoc();
            OpBuilder builder(op);

            Value matrix = op->getResult(0);
            auto rows = matType.getNumRows();
            auto cols = matType.getNumCols();

            Value totalElements = builder.create<arith::ConstantIndexOp>(loc, rows * cols);
            Value zeroCount = builder.create<arith::ConstantIndexOp>(loc, 0);

            Value zeroF32 = builder.create<arith::ConstantOp>(loc, builder.getF32Type(), builder.getF32FloatAttr(0.0));
            Value oneF32 = builder.create<arith::ConstantOp>(loc, builder.getF32Type(), builder.getF32FloatAttr(1.0));

            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    auto element = builder.create<memref::LoadOp>(loc, matrix, ValueRange{builder.create<arith::ConstantIndexOp>(loc, i), builder.create<arith::ConstantIndexOp>(loc, j)});
                    auto isZero = builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, element, zeroF32);
                    auto zeroCountIfZero = builder.create<arith::SelectOp>(loc, isZero, oneF32, zeroF32);
                    zeroCount = builder.create<arith::AddFOp>(loc, zeroCount, zeroCountIfZero);
                }
            }

            auto zeroElementsF32 = builder.create<arith::UIToFPOp>(loc, zeroCount, builder.getF32Type());
            auto totalElementsF32 = builder.create<arith::UIToFPOp>(loc, totalElements, builder.getF32Type());
            auto sparsity = builder.create<arith::DivFOp>(loc, zeroElementsF32, totalElementsF32);

            // Store the sparsity in the results JSON
            std::string opName = op->getName().getStringRef().str();
            double sparsityValue = sparsity.getType().cast<arith::ConstantOp>().value().convertToFloat();
            if (sparsityResults.find(opName) == sparsityResults.end()) {
                sparsityResults[opName] = json::array();
            }
            sparsityResults[opName].push_back(sparsityValue);

            op->getResult(0).replaceAllUsesWith(sparsity);
        }

    public:
        CalculateSparsityPass() {}

        void runOnOperation() override {
            std::cout << "Starting CalculateSparsityPass on function: " << getOperation().getName().str() << std::endl;
            func::FuncOp f = getOperation();
            f.walk<WalkOrder::PreOrder>(walkOp);

            // Write the results to a JSON file
            std::ofstream file("sparsity_results.json");
            file << sparsityResults.dump(4); // Pretty print with 4 spaces
            file.close();
            std::cout << "Finished CalculateSparsityPass on function: " << getOperation().getName().str() << std::endl;
        }

        StringRef getArgument() const final { return "calculate-sparsity"; }
        StringRef getDescription() const final { return "Calculate the sparsity of matrix operations"; }
    };
}

std::unique_ptr<Pass> daphne::createCalculateSparsityPass() {
    return std::make_unique<CalculateSparsityPass>();
}