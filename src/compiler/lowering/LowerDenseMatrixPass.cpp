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
#include "llvm/ADT/APFloat.h"

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

        auto tensorType = tensor.getElementType();
        auto memRefType = mlir::MemRefType::get(
            {nR, nC}, tensorType);
        auto memRefShape = memRefType.getShape();
        auto memRef = rewriter.create<mlir::daphne::GetMemRefDenseMatrix>(
            op->getLoc(), memRefType, operands[0]);

        llvm::APFloat zero = tensorType.isF32() ? llvm::APFloat(float(0)) : llvm::APFloat(0.0);
        Value sum = rewriter.create<mlir::ConstantFloatOp>(
            op->getLoc(), zero, tensorType.dyn_cast<mlir::FloatType>());

        SmallVector<Value, 4> loopIvs;
        // SmallVector<scf::ForOp, 2> forOps;
        SmallVector<AffineForOp, 2> forOps;
        // auto lowerBound = rewriter.create<ConstantIndexOp>(loc, 0);
        // auto outerUpperBound =
        //     rewriter.create<ConstantIndexOp>(loc, memRefShape[0]);
        // auto step = rewriter.create<ConstantIndexOp>(loc, 1);
        // outer loop
        // auto outerLoop = rewriter.create<scf::ForOp>(
        auto outerLoop = rewriter.create<AffineForOp>(
            loc, 0, nR, 1, ValueRange{sum});
        for (Operation &nested : *outerLoop.getBody()) {
            rewriter.eraseOp(&nested);
        }
        loopIvs.push_back(outerLoop.getInductionVar());
        // outer loop body
        rewriter.setInsertionPointToStart(outerLoop.getBody());
        Value sum_iter = rewriter.create<mlir::ConstantFloatOp>(
            op->getLoc(), zero,
            tensorType.dyn_cast<mlir::FloatType>());
        // inner loop
        // auto innerUpperBound =
        //     rewriter.create<ConstantIndexOp>(loc, memRefShape[1]);
        // auto innerLoop = rewriter.create<scf::ForOp>(
        auto innerLoop = rewriter.create<AffineForOp>(
            loc, 0, nC, 1, ValueRange{sum_iter});
        for (Operation &nested : *innerLoop.getBody()) {
            rewriter.eraseOp(&nested);
        }
        loopIvs.push_back(innerLoop.getInductionVar());
        // inner loop body
        rewriter.setInsertionPointToStart(innerLoop.getBody());
        // load value from memref
        auto elementLoad =
            rewriter.create<memref::LoadOp>(loc, memRef, loopIvs);
        // sum loop iter arg and memref value
        mlir::Value inner_sum = rewriter.create<AddFOp>(
            loc, innerLoop.getRegionIterArgs()[0], elementLoad);
        // yield inner loop result
        rewriter.setInsertionPointToEnd(innerLoop.getBody());
        // rewriter.create<scf::YieldOp>(loc, inner_sum);
        rewriter.create<AffineYieldOp>(loc, inner_sum);
        // yield outer loop result
        rewriter.setInsertionPointToEnd(outerLoop.getBody());
        mlir::Value outer_sum = rewriter.create<AddFOp>(
            loc, outerLoop.getRegionIterArgs()[0], innerLoop.getResult(0));
        // rewriter.create<scf::YieldOp>(loc, outer_sum);
        rewriter.create<AffineYieldOp>(loc, outer_sum);

        // replace sumAll op with result of loops
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
    target.addLegalDialect<mlir::AffineDialect>();

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
