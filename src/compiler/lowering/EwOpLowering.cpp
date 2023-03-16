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
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

class EwModOpLowering
    : public mlir::OpConversionPattern<mlir::daphne::EwModOp> {
   public:
    using OpConversionPattern::OpConversionPattern;

    // TODO(phil): currently only supports f64
    mlir::LogicalResult matchAndRewrite(
        mlir::daphne::EwModOp op, OpAdaptor adaptor,
        mlir::ConversionPatternRewriter &rewriter) const override {
        auto loc = op->getLoc();
        mlir::daphne::MatrixType lhsTensor =
            adaptor.getLhs().getType().dyn_cast<mlir::daphne::MatrixType>();

        auto tensorType = lhsTensor.getElementType();
        if (!tensorType.isF64()) return failure();

        auto lhsRows = lhsTensor.getNumRows();
        auto lhsCols = lhsTensor.getNumCols();

        auto lhsMemRefType =
            mlir::MemRefType::get({lhsRows, lhsCols}, tensorType);

        mlir::MemRefType outputMemRefType =
            mlir::MemRefType::get({lhsCols, lhsRows}, tensorType);

        // daphne::Matrix -> memref
        mlir::Value lhs = rewriter.create<mlir::daphne::GetMemRefDenseMatrix>(
            op->getLoc(), lhsMemRefType, adaptor.getLhs());
        mlir::Value rhs = adaptor.getRhs();
        std::pair<bool, double> isConstant =
            CompilerUtils::isConstant<double>(rhs);

        // Apply (n & (n - 1)) optimization when n is a power of two
        bool optimize =
            isConstant.first && std::fmod(isConstant.second, 2) == 0;

        mlir::Value cst_one{};
        mlir::Value rhsValue{};
        mlir::Value rhsV{};

        if (optimize) {
            cst_one = rewriter.create<mlir::arith::ConstantOp>(
                loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(1));
            rhsValue = rewriter.create<mlir::arith::SubFOp>(loc, rhs, cst_one);

            rhsV = rewriter.create<mlir::arith::FPToSIOp>(
                loc, rewriter.getI64Type(), rhsValue);
        } else {
            rhsV = rhs;
        }

        // Alloc output memref
        mlir::Value outputMemRef =
            rewriter.create<mlir::memref::AllocOp>(loc, outputMemRefType);

        // Fill the output MemRef
        affineFillMemRef(0.0, rewriter, loc, outputMemRefType.getShape(),
                         op->getContext(), outputMemRef, tensorType);

        SmallVector<Value, 4> loopIvs;

        auto outerLoop = rewriter.create<AffineForOp>(loc, 0, lhsRows, 1);
        for (Operation &nested : *outerLoop.getBody()) {
            rewriter.eraseOp(&nested);
        }
        loopIvs.push_back(outerLoop.getInductionVar());

        // outer loop body
        rewriter.setInsertionPointToStart(outerLoop.getBody());
        auto innerLoop = rewriter.create<AffineForOp>(loc, 0, lhsCols, 1);
        for (Operation &nested : *innerLoop.getBody()) {
            rewriter.eraseOp(&nested);
        }
        loopIvs.push_back(innerLoop.getInductionVar());
        rewriter.create<AffineYieldOp>(loc);
        rewriter.setInsertionPointToStart(innerLoop.getBody());
        mlir::Value lhsValue = rewriter.create<AffineLoadOp>(loc, lhs, loopIvs);

        mlir::Value modResult{};
        if (optimize) {
            mlir::Value lhsV = rewriter.create<mlir::arith::FPToSIOp>(
                loc, rewriter.getI64Type(), lhsValue);
            mlir::Value modResultCast =
                rewriter.create<arith::AndIOp>(loc, lhsV, rhsV);
            modResult = rewriter.create<arith::SIToFPOp>(
                loc, rewriter.getF64Type(), modResultCast);
        } else {
            modResult = rewriter.create<arith::RemFOp>(loc, lhsValue, rhsV);
        }

        rewriter.create<AffineStoreOp>(loc, modResult, outputMemRef, loopIvs);

        rewriter.create<AffineYieldOp>(loc);
        rewriter.setInsertionPointAfter(outerLoop);

        auto extractStridedMetadataOp =
            rewriter.create<memref::ExtractStridedMetadataOp>(loc,
                                                              outputMemRef);
        // aligned ptr (memref.data)
        mlir::Value alignedPtr =
            rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
                loc, outputMemRef);
        // offset
        mlir::Value offset = extractStridedMetadataOp.getOffset();
        // strides
        mlir::ResultRange strides = extractStridedMetadataOp.getStrides();
        // sizes
        mlir::ResultRange sizes = extractStridedMetadataOp.getSizes();

        mlir::Value DM =
            rewriter.create<mlir::daphne::GetDenseMatrixFromMemRef>(
                loc, op.getType(), alignedPtr, offset, sizes[0], sizes[1],
                strides[0], strides[1]);
        rewriter.replaceOp(op, DM);
        return mlir::success();
    }
};

namespace {
struct EwOpLoweringPass
    : public mlir::PassWrapper<EwOpLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
    explicit EwOpLoweringPass() {}

    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry
            .insert<mlir::LLVM::LLVMDialect, mlir::AffineDialect,
                    mlir::memref::MemRefDialect, mlir::daphne::DaphneDialect>();
    }
    void runOnOperation() final;
};
}  // end anonymous namespace

void EwOpLoweringPass::runOnOperation() {
    mlir::ConversionTarget target(getContext());
    mlir::RewritePatternSet patterns(&getContext());
    mlir::LowerToLLVMOptions llvmOptions(&getContext());
    mlir::LLVMTypeConverter typeConverter(&getContext(), llvmOptions);

    target
        .addLegalDialect<mlir::AffineDialect, arith::ArithDialect,
                         memref::MemRefDialect, mlir::daphne::DaphneDialect>();

    target.addIllegalOp<mlir::daphne::EwModOp>();

    patterns.insert<EwModOpLowering>(&getContext());
    auto module = getOperation();
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<mlir::Pass> mlir::daphne::createEwOpLoweringPass() {
    return std::make_unique<EwOpLoweringPass>();
}
