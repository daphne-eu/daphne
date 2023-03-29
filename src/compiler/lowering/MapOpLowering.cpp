/*
 * Copyright 2023 The DAPHNE Consortium
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

#include "compiler/utils/CompilerUtils.h"
#include "compiler/utils/LoweringUtils.h"
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

class InlineMapOpLowering
    : public mlir::OpConversionPattern<mlir::daphne::MapOp> {
   public:
    using OpConversionPattern::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        mlir::daphne::MapOp op, OpAdaptor adaptor,
        mlir::ConversionPatternRewriter &rewriter) const override {
        auto loc = op->getLoc();

        // TODO(phil): why doesn't adaptor.getArg().getType() work?
        mlir::daphne::MatrixType lhsTensor =
            op->getOperandTypes().front().dyn_cast<mlir::daphne::MatrixType>();
        auto tensorType = lhsTensor.getElementType();
        auto lhsMemRefType = mlir::MemRefType::get(
            {lhsTensor.getNumRows(), lhsTensor.getNumCols()}, tensorType);

        mlir::Value lhs = rewriter.create<mlir::daphne::GetMemRefDenseMatrix>(
            loc, lhsMemRefType, adaptor.getArg());

        // auto module = op->getParentOp()->getParentOfType<mlir::ModuleOp>();
        // mlir::ModuleOp modOp = module.dyn_cast<mlir::ModuleOp>();
        // LLVM::LLVMFuncOp udfFuncOp =
        // module.lookupSymbol<LLVM::LLVMFuncOp>(op.getFunc());
        // udfFuncOp.getArgumentTypes()[0].dump();
        // auto region = udfFuncOp.getCallableRegion();

        mlir::Value cst_one = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(1.0));
        SmallVector<int64_t, 4> lowerBounds(/*Rank=*/2, /*Value=*/0);
        SmallVector<int64_t, 4> steps(/*Rank=*/2, /*Value=*/1);
        buildAffineLoopNest(
            rewriter, op.getLoc(), lowerBounds,
            {lhsTensor.getNumRows(), lhsTensor.getNumCols()}, steps,
            [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
                // Call the processing function with the rewriter, the memref
                // operands, and the loop induction variables. This function
                // will return the value to store at the current index.
                mlir::Value load =
                    nestedBuilder.create<AffineLoadOp>(loc, lhs, ivs);
                mlir::Value add =
                    nestedBuilder.create<arith::AddFOp>(loc, load, cst_one);
                nestedBuilder.create<AffineStoreOp>(loc, add, lhs, ivs);
            });
        mlir::Value output =
            getDenseMatrixFromMemRef(op->getLoc(), rewriter, lhs, op.getType());
        rewriter.replaceOp(op, output);
        return mlir::success();
    }
};

namespace {
struct MapOpLoweringPass
    : public mlir::PassWrapper<MapOpLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
    explicit MapOpLoweringPass() {}

    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry
            .insert<mlir::LLVM::LLVMDialect, mlir::AffineDialect,
                    mlir::memref::MemRefDialect, mlir::daphne::DaphneDialect>();
    }
    void runOnOperation() final;
};
}  // end anonymous namespace

void MapOpLoweringPass::runOnOperation() {
    mlir::ConversionTarget target(getContext());
    mlir::RewritePatternSet patterns(&getContext());
    mlir::LowerToLLVMOptions llvmOptions(&getContext());
    mlir::LLVMTypeConverter typeConverter(&getContext(), llvmOptions);

    target
        .addLegalDialect<mlir::AffineDialect, arith::ArithDialect,
                         memref::MemRefDialect, mlir::daphne::DaphneDialect>();

    target.addIllegalOp<mlir::daphne::MapOp>();

    patterns.insert<InlineMapOpLowering>(&getContext());
    auto module = getOperation();
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<mlir::Pass> mlir::daphne::createMapOpLoweringPass() {
    return std::make_unique<MapOpLoweringPass>();
}
