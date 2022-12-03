/*
 * Copyright 2022 The DAPHNE Consortium
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

#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "compiler/utils/CompilerUtils.h"
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

class FillOpLowering : public OpConversionPattern<daphne::FillOp> {
   public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        daphne::FillOp op, ArrayRef<Value> operands,
        ConversionPatternRewriter &rewriter) const override {
        return success();
    }
};

class SumAllOpLowering : public OpConversionPattern<daphne::AllAggSumOp> {
   public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        daphne::AllAggSumOp op, ArrayRef<Value> operands,
        ConversionPatternRewriter &rewriter) const override {
        mlir::daphne::MatrixType tensor =
            operands[0].getType().dyn_cast<mlir::daphne::MatrixType>();

        auto loc = op->getLoc();
        auto nR = tensor.getNumRows();
        auto nC = tensor.getNumCols();

        auto memRefType = mlir::MemRefType::get(
            {nR, nC}, mlir::Float64Type::get(op->getContext()));
        auto memRefShape = memRefType.getShape();
        auto memRef = rewriter.create<mlir::daphne::GetMemRefDenseMatrix>(
            op->getLoc(), memRefType, operands[0]);

        Value sum = rewriter.create<mlir::ConstantFloatOp>(
            op->getLoc(), llvm::APFloat(0.0),
            Float64Type::get(op->getContext()));

        SmallVector<Value, 4> loopIvs;
        SmallVector<scf::ForOp, 2> forOps;
        auto lowerBound = rewriter.create<ConstantIndexOp>(loc, 0);
        auto outerUpperBound =
            rewriter.create<ConstantIndexOp>(loc, memRefShape[0]);
        auto step = rewriter.create<ConstantIndexOp>(loc, 1);
        // outer loop
        auto outerLoop = rewriter.create<scf::ForOp>(
            loc, lowerBound, outerUpperBound, step, ValueRange{sum});
        for (Operation &nested : *outerLoop.getBody()) {
            rewriter.eraseOp(&nested);
        }
        loopIvs.push_back(outerLoop.getInductionVar());
        // outer loop body
        rewriter.setInsertionPointToStart(outerLoop.getBody());
        Value sum_iter = rewriter.create<mlir::ConstantFloatOp>(
            op->getLoc(), llvm::APFloat(0.0),
            Float64Type::get(op->getContext()));
        // // inner loop
        auto innerUpperBound =
            rewriter.create<ConstantIndexOp>(loc, memRefShape[1]);
        auto innerLoop = rewriter.create<scf::ForOp>(
            loc, lowerBound, innerUpperBound, step, ValueRange{sum_iter});
        for (Operation &nested : *innerLoop.getBody()) {
            rewriter.eraseOp(&nested);
        }
        loopIvs.push_back(innerLoop.getInductionVar());
        // inner loop body
        rewriter.setInsertionPointToStart(innerLoop.getBody());
        // try loaded with ConstantIndexOp [{0,0}]
        auto elementLoad =
            rewriter.create<memref::LoadOp>(loc, memRef, loopIvs);
        mlir::Value inner_sum = rewriter.create<AddFOp>(
            loc, innerLoop.getRegionIterArgs()[0], elementLoad);
        // yield inner loop result
        rewriter.setInsertionPointToEnd(innerLoop.getBody());
        rewriter.create<scf::YieldOp>(loc, inner_sum);
        // yield outer loop result
        rewriter.setInsertionPointToEnd(outerLoop.getBody());
        mlir::Value outer_sum = rewriter.create<AddFOp>(
            loc, outerLoop.getRegionIterArgs()[0], innerLoop.getResult(0));
        rewriter.create<scf::YieldOp>(loc, outer_sum);

        // for (size_t i = 0; i < memRefShape.size(); ++i) {
        // auto lowerBound = rewriter.create<ConstantIndexOp>(loc, 0);
        //     auto upperBound =
        //         rewriter.create<ConstantIndexOp>(loc, memRefShape[i]);
        //     auto step = rewriter.create<ConstantIndexOp>(loc, 1);
        //     auto loop =
        //         rewriter.create<scf::ForOp>(loc, lowerBound, upperBound,
        //         step, ValueRange{sum});
        //     forOps.push_back(loop);
        //     // for (Operation &nested : *loop.getBody()) {
        //     //     rewriter.eraseOp(&nested);
        //     // }
        //     loopIvs.push_back(loop.getInductionVar());
        //     // outer loop body
        //     rewriter.setInsertionPointToStart(loop.getBody());
        //     Value sum_iter = rewriter.create<mlir::ConstantFloatOp>(
        //         op->getLoc(), llvm::APFloat(0.0),
        //         Float64Type::get(op->getContext()));
        //     rewriter.setInsertionPointToEnd(loop.getBody());
        //     rewriter.create<scf::YieldOp>(loc);
        //     rewriter.setInsertionPointToStart(loop.getBody());
        // }
        // inner loop body
        // auto elementLoad =
        //     rewriter.create<memref::LoadOp>(loc, memRef, loopIvs);
        // sum= rewriter.create<AddFOp>(loc, sum, elementLoad);
        // sum_iter = rewriter.create<AddFOp>(loc, sum_iter, elementLoad);
        // rewriter.create<scf::YieldOp>(loc, sum_iter);
        // rewriter.replaceOpWithNewOp<mlir::daphne::GetDataPointer>(
        //     op.getOperation(), rewriter.getIntegerType(64, false),
        //     operands[0]);

        // rewriter.setInsertionPoint(op);
        // SmallVector<Value, 4> idxs;
        // idxs.push_back(rewriter.create<ConstantIndexOp>(loc, 0));
        // idxs.push_back(rewriter.create<ConstantIndexOp>(loc, 0));
        // mlir::Value load = rewriter.create<memref::LoadOp>(loc, memRef,
        // idxs);
        rewriter.replaceOp(op, outerLoop.getResult(0));
        return success();
    }
};

namespace {
struct LowerDenseMatrixPass
    : public mlir::PassWrapper<LowerDenseMatrixPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
    explicit LowerDenseMatrixPass() {}

    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<mlir::LLVM::LLVMDialect>();
    }
    void runOnOperation() final;
};
}  // end anonymous namespace

void LowerDenseMatrixPass::runOnOperation() {
    mlir::ConversionTarget target(getContext());
    mlir::OwningRewritePatternList patterns(&getContext());
    LowerToLLVMOptions llvmOptions(&getContext());
    llvmOptions.emitCWrappers = true;
    LLVMTypeConverter typeConverter(&getContext(), llvmOptions);

    target.addLegalDialect<mlir::memref::MemRefDialect>();
    target.addLegalDialect<mlir::StandardOpsDialect>();
    target.addLegalDialect<mlir::scf::SCFDialect>();

    target.addLegalOp<mlir::daphne::GetMemRefDenseMatrix>();
    target.addIllegalOp<mlir::daphne::AllAggSumOp>();

    typeConverter.addConversion([&](daphne::MatrixType t) {
        return mlir::MemRefType::get({t.getNumRows(), t.getNumCols()},
                                     t.getElementType());
    });

    patterns.insert<SumAllOpLowering>(&getContext());
    auto module = getOperation();
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<mlir::Pass> mlir::daphne::createLowerDenseMatrixPass() {
    return std::make_unique<LowerDenseMatrixPass>();
}
