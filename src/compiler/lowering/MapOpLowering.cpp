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
#include "mlir/Dialect/Func/IR/FuncOps.h"
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

        mlir::daphne::MatrixType lhsMatrixType =
            op->getOperandTypes().front().dyn_cast<mlir::daphne::MatrixType>();
        auto matrixElementType = lhsMatrixType.getElementType();
        auto lhsMemRefType = mlir::MemRefType::get(
            {lhsMatrixType.getNumRows(), lhsMatrixType.getNumCols()}, matrixElementType);

        mlir::Value lhs =
            rewriter.create<mlir::daphne::ConvertDenseMatrixToMemRef>(
                loc, lhsMemRefType, adaptor.getArg());
        mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
        func::FuncOp udfFuncOp =
            module.lookupSymbol<func::FuncOp>(op.getFunc());

        SmallVector<Value, 4> loopIvs;

        auto outerLoop =
            rewriter.create<AffineForOp>(loc, 0, lhsMatrixType.getNumRows(), 1);
        for (Operation &nested : *outerLoop.getBody()) {
            rewriter.eraseOp(&nested);
        }
        loopIvs.push_back(outerLoop.getInductionVar());

        // outer loop body
        rewriter.setInsertionPointToStart(outerLoop.getBody());
        auto innerLoop =
            rewriter.create<AffineForOp>(loc, 0, lhsMatrixType.getNumCols(), 1);
        for (Operation &nested : *innerLoop.getBody()) {
            rewriter.eraseOp(&nested);
        }
        loopIvs.push_back(innerLoop.getInductionVar());
        rewriter.create<AffineYieldOp>(loc);
        rewriter.setInsertionPointToStart(innerLoop.getBody());

        // inner loop body
        mlir::Value lhsValue = rewriter.create<AffineLoadOp>(loc, lhs, loopIvs);
        mlir::Value res =
            rewriter.create<func::CallOp>(loc, udfFuncOp, ValueRange{lhsValue})
                ->getResult(0);
        rewriter.create<AffineStoreOp>(loc, res, lhs, loopIvs);
        rewriter.create<AffineYieldOp>(loc);

        rewriter.setInsertionPointAfter(outerLoop);
        mlir::Value output = convertMemRefToDenseMatrix(op->getLoc(), rewriter,
                                                        lhs, op.getType());
        rewriter.replaceOp(op, output);
        return mlir::success();
    }
};

namespace {
/**
 * @brief The MapOpLoweringPass rewrites the daphne::MapOp operator
 * to a set of perfectly nested affine loops and inserts for each element a call
 * to the UDF assigned to the daphne::MapOp.
 *
 * This rewrite enables subsequent inlining pass to completely replace
 * the daphne::MapOp by inlining the produced CallOps from this pass.
 */
struct MapOpLoweringPass
    : public mlir::PassWrapper<MapOpLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
    explicit MapOpLoweringPass() {}

    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<mlir::LLVM::LLVMDialect, mlir::AffineDialect,
                        mlir::memref::MemRefDialect,
                        mlir::daphne::DaphneDialect, mlir::func::FuncDialect>();
    }
    void runOnOperation() final;

    StringRef getArgument() const final { return "lower-map"; }
    StringRef getDescription() const final {
        return "Lowers the daphne.mapOp operation to"
               "a set of affine loops, directly calling the UDF. "
               "Subsequent use of the inlining pass may inline the call to the "
               "UDF.";
    }
};
}  // end anonymous namespace

void MapOpLoweringPass::runOnOperation() {
    mlir::ConversionTarget target(getContext());
    mlir::RewritePatternSet patterns(&getContext());
    mlir::LowerToLLVMOptions llvmOptions(&getContext());
    mlir::LLVMTypeConverter typeConverter(&getContext(), llvmOptions);

    target.addLegalDialect<mlir::AffineDialect, arith::ArithDialect,
                           memref::MemRefDialect, mlir::daphne::DaphneDialect,
                           mlir::func::FuncDialect>();

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
