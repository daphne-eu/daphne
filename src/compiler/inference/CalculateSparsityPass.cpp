#include <util/ErrorHandler.h>
#include <ir/daphneir/Daphne.h>
#include <ir/daphneir/Passes.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Operation.h>
#include <mlir/Pass/Pass.h>

#include <stdexcept>
#include <memory>
#include <vector>
#include <utility>

using namespace mlir;

namespace {
    class CalculateSparsityPass : public PassWrapper<CalculateSparsityPass, OperationPass<func::FuncOp>> {

        std::function<WalkResult(Operation*)> walkOp = [&](Operation * op) {
            if(auto matType = op->getResult(0).getType().dyn_cast<daphne::MatrixType>()) {
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

            op->getResult(0).replaceAllUsesWith(sparsity);
        }

    public:
        CalculateSparsityPass() {}

        void runOnOperation() override {
            func::FuncOp f = getOperation();
            f.walk<WalkOrder::PreOrder>(walkOp);
        }

        StringRef getArgument() const final { return "calculate-sparsity"; }
        StringRef getDescription() const final { return "Calculate the sparsity of matrix operations"; }
    };
}

std::unique_ptr<Pass> daphne::createCalculateSparsityPass() {
    return std::make_unique<CalculateSparsityPass>();
}

static PassRegistration<CalculateSparsityPass> pass("calculate-sparsity", "Calculate the sparsity of matrix operations");