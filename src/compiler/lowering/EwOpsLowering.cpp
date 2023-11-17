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

#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "compiler/utils/CompilerUtils.h"
#include "compiler/utils/LoweringUtils.h"
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

template <class UnaryOp, class IOp, class FOp>
struct UnaryOpLowering : public mlir::OpConversionPattern<UnaryOp> {
    using OpAdaptor = typename mlir::OpConversionPattern<UnaryOp>::OpAdaptor;

   public:
    UnaryOpLowering(mlir::TypeConverter &typeConverter, mlir::MLIRContext *ctx)
        : mlir::OpConversionPattern<UnaryOp>(typeConverter, ctx) {
        this->setDebugName("EwDaphneOpsLowering");
    }

    mlir::LogicalResult matchAndRewrite(
        UnaryOp op, OpAdaptor adaptor,
        mlir::ConversionPatternRewriter &rewriter) const override {
        mlir::Type type = op.getType();

        if (type.isa<mlir::IntegerType>()) {
            rewriter.replaceOpWithNewOp<IOp>(op.getOperation(),
                                             adaptor.getOperands());
        } else if (type.isa<mlir::FloatType>()) {
            rewriter.replaceOpWithNewOp<FOp>(op.getOperation(),
                                             adaptor.getOperands());
        } else {
            return mlir::failure();
        }
        return mlir::success();
    }
};

template <class BinaryOp, class IOp, class FOp>
class BinaryOpLowering final : public mlir::OpConversionPattern<BinaryOp> {
    using OpAdaptor = typename mlir::OpConversionPattern<BinaryOp>::OpAdaptor;

   public:
    BinaryOpLowering(mlir::TypeConverter &typeConverter, mlir::MLIRContext *ctx)
        : mlir::OpConversionPattern<BinaryOp>(typeConverter, ctx) {
        this->setDebugName("EwDaphneOpLowering");
    }

    mlir::LogicalResult convertEwScalar(
        BinaryOp op, OpAdaptor adaptor,
        mlir::ConversionPatternRewriter &rewriter) const {
        auto lhs = adaptor.getLhs();
        auto rhs = adaptor.getRhs();
        auto loc = op.getLoc();

        if (lhs.getType().template isa<mlir::FloatType>() &&
            rhs.getType().template isa<mlir::FloatType>()) {
            rewriter.replaceOpWithNewOp<FOp>(op.getOperation(),
                                             adaptor.getOperands());
            return mlir::success();
        }

        Value castedLhs = this->typeConverter->materializeTargetConversion(
            rewriter, loc,
            rewriter.getIntegerType(
                adaptor.getRhs().getType().getIntOrFloatBitWidth()),
            ValueRange{adaptor.getLhs()});

        Value castedRhs = this->typeConverter->materializeTargetConversion(
            rewriter, loc,
            rewriter.getIntegerType(
                adaptor.getRhs().getType().getIntOrFloatBitWidth()),
            ValueRange{adaptor.getRhs()});

        Value binaryOp = rewriter.create<IOp>(loc, castedLhs, castedRhs);

        Value res = this->typeConverter->materializeSourceConversion(
            rewriter, loc, lhs.getType(), ValueRange{binaryOp});

        rewriter.replaceOp(op, res);
        return mlir::success();
    }

    mlir::LogicalResult matchAndRewrite(
        BinaryOp op, OpAdaptor adaptor,
        mlir::ConversionPatternRewriter &rewriter) const override {
        auto lhs = adaptor.getLhs();
        auto rhs = adaptor.getRhs();

        // no matrix
        if (!lhs.getType().template isa<mlir::daphne::MatrixType>() &&
            !rhs.getType().template isa<mlir::daphne::MatrixType>())
            return convertEwScalar(op, adaptor, rewriter);

        // for now assume matrix is LHS and RHS is non matrix
        mlir::daphne::MatrixType lhsMatrixType =
            adaptor.getLhs()
                .getType()
                .template dyn_cast<mlir::daphne::MatrixType>();
        auto matrixElementType = lhsMatrixType.getElementType();
        auto lhsRows = lhsMatrixType.getNumRows();
        auto lhsCols = lhsMatrixType.getNumCols();
        auto lhsMemRefType =
            mlir::MemRefType::get({lhsRows, lhsCols}, matrixElementType);

        mlir::Type elementType{};
        mlir::Value memRefLhs =
            rewriter.create<mlir::daphne::ConvertDenseMatrixToMemRef>(
                op->getLoc(), lhsMemRefType, adaptor.getLhs());

        mlir::Value memRefRhs{};
        bool isMatrixMatrix =
            rhs.getType().template isa<mlir::daphne::MatrixType>();

        if (isMatrixMatrix) {
            memRefRhs =
                rewriter.create<mlir::daphne::ConvertDenseMatrixToMemRef>(
                    op->getLoc(), lhsMemRefType, adaptor.getRhs());
            elementType = lhsMemRefType.getElementType();
        } else {
            elementType = rhs.getType();
        }

        mlir::Value outputMemRef =
            insertMemRefAlloc(lhsMemRefType, op->getLoc(), rewriter);

        SmallVector<int64_t, 4> lowerBounds(/*Rank=*/2, /*Value=*/0);
        SmallVector<int64_t, 4> steps(/*Rank=*/2, /*Value=*/1);
        buildAffineLoopNest(
            rewriter, op.getLoc(), lowerBounds,
            {lhsMatrixType.getNumRows(), lhsMatrixType.getNumCols()}, steps,
            [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
                mlir::Value loadLhs =
                    nestedBuilder.create<AffineLoadOp>(loc, memRefLhs, ivs);
                mlir::Value binaryOp{};

                if (adaptor.getRhs()
                        .getType()
                        .template isa<mlir::FloatType>()) {
                    binaryOp = nestedBuilder.create<FOp>(loc, loadLhs,
                                                         adaptor.getRhs());

                    nestedBuilder.create<AffineStoreOp>(loc, binaryOp,
                                                        outputMemRef, ivs);
                    return;
                }

                mlir::Value rhs{};
                if (isMatrixMatrix)
                    rhs =
                        nestedBuilder.create<AffineLoadOp>(loc, memRefRhs, ivs);
                else
                    rhs = adaptor.getRhs();

                // is integer
                if (elementType.isInteger(
                        elementType.getIntOrFloatBitWidth())) {
                    Value castedLhs =
                        this->typeConverter->materializeTargetConversion(
                            nestedBuilder, loc,
                            nestedBuilder.getIntegerType(
                                lhsMemRefType.getElementTypeBitWidth()),
                            ValueRange{loadLhs});

                    Value castedRhs =
                        this->typeConverter->materializeTargetConversion(
                            nestedBuilder, loc,
                            nestedBuilder.getIntegerType(
                                lhsMemRefType.getElementTypeBitWidth()),
                            ValueRange{rhs});

                    binaryOp =
                        nestedBuilder.create<IOp>(loc, castedLhs, castedRhs);
                    Value castedRes =
                        this->typeConverter->materializeSourceConversion(
                            nestedBuilder, loc, elementType,
                            ValueRange{binaryOp});
                    nestedBuilder.create<AffineStoreOp>(loc, castedRes,
                                                        outputMemRef, ivs);
                } else {
                    // is float
                    binaryOp = nestedBuilder.create<FOp>(loc, loadLhs, rhs);
                    nestedBuilder.create<AffineStoreOp>(loc, binaryOp,
                                                        outputMemRef, ivs);
                }
            });
        mlir::Value output = convertMemRefToDenseMatrix(
            op->getLoc(), rewriter, outputMemRef, op.getType());

        rewriter.replaceOp(op, output);
        return mlir::success();
    }
};

// clang-format off
// math::sqrt only supports floating point, DAPHNE promotes argument type of sqrt to f32/64
using SqrtOpLowering = UnaryOpLowering<mlir::daphne::EwSqrtOp, mlir::math::SqrtOp, mlir::math::SqrtOp>;
using AbsOpLowering = UnaryOpLowering<mlir::daphne::EwAbsOp, mlir::math::AbsIOp, mlir::math::AbsFOp>;
using AddOpLowering = BinaryOpLowering<mlir::daphne::EwAddOp, mlir::arith::AddIOp, mlir::arith::AddFOp>;
using SubOpLowering = BinaryOpLowering<mlir::daphne::EwSubOp, mlir::arith::SubIOp, mlir::arith::SubFOp>;
using MulOpLowering = BinaryOpLowering<mlir::daphne::EwMulOp, mlir::arith::MulIOp, mlir::arith::MulFOp>;
using DivOpLowering = BinaryOpLowering<mlir::daphne::EwDivOp, mlir::arith::DivSIOp, mlir::arith::DivFOp>;
using PowOpLowering = BinaryOpLowering<mlir::daphne::EwPowOp, mlir::math::PowFOp, mlir::math::PowFOp>;
// clang-format on

namespace {
/**
 * @brief This pass lowers element-wise operations to affine loop
 * structures and arithmetic operations.
 *
 * This rewrite may enable loop fusion of the produced affine loops by
 * running the loop fusion pass.
 */
struct EwOpLoweringPass
    : public mlir::PassWrapper<EwOpLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
    explicit EwOpLoweringPass() {}

    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<mlir::LLVM::LLVMDialect, mlir::AffineDialect,
                        mlir::memref::MemRefDialect,
                        mlir::daphne::DaphneDialect, mlir::math::MathDialect>();
    }
    void runOnOperation() final;

    StringRef getArgument() const final { return "lower-ew"; }
    StringRef getDescription() const final {
        return "This pass lowers element-wise operations to affine-loop "
               "structures and arithmetic operations.";
    }
};
}  // end anonymous namespace

void populateLowerEwOpConversionPatterns(mlir::LLVMTypeConverter &typeConverter,
                                         mlir::RewritePatternSet &patterns) {
    // clang-format off
    patterns.insert<
        AddOpLowering,
        SubOpLowering,
        MulOpLowering,
        SqrtOpLowering,
        AbsOpLowering,
        DivOpLowering,
        PowOpLowering>(typeConverter, patterns.getContext());
    // clang-format on
}

void EwOpLoweringPass::runOnOperation() {
    mlir::ConversionTarget target(getContext());
    mlir::RewritePatternSet patterns(&getContext());
    mlir::LowerToLLVMOptions llvmOptions(&getContext());
    mlir::LLVMTypeConverter typeConverter(&getContext(), llvmOptions);

    typeConverter.addConversion(convertInteger);
    typeConverter.addConversion(convertFloat);
    typeConverter.addConversion([](Type type) { return type; });
    typeConverter.addArgumentMaterialization(materializeCastFromIllegal);
    typeConverter.addSourceMaterialization(materializeCastToIllegal);
    typeConverter.addTargetMaterialization(materializeCastFromIllegal);

    target.addLegalDialect<mlir::arith::ArithDialect,
                           mlir::memref::MemRefDialect, mlir::AffineDialect,
                           mlir::LLVM::LLVMDialect, mlir::daphne::DaphneDialect,
                           mlir::BuiltinDialect, mlir::math::MathDialect>();

    target.addDynamicallyLegalOp<mlir::daphne::EwSqrtOp, mlir::daphne::EwAbsOp>(
        [](Operation *op) {
            return op->getOperandTypes()[0].isa<mlir::daphne::MatrixType>();
        });

    target.addDynamicallyLegalOp<mlir::daphne::EwAddOp, mlir::daphne::EwSubOp,
                                 mlir::daphne::EwMulOp, mlir::daphne::EwPowOp,
                                 mlir::daphne::EwDivOp>([](Operation *op) {
        if (op->getOperandTypes()[0].isa<mlir::daphne::MatrixType>() &&
            op->getOperandTypes()[1].isa<mlir::daphne::MatrixType>()) {
            mlir::daphne::MatrixType lhs =
                op->getOperandTypes()[0]
                    .template dyn_cast<mlir::daphne::MatrixType>();
            mlir::daphne::MatrixType rhs =
                op->getOperandTypes()[1]
                    .template dyn_cast<mlir::daphne::MatrixType>();
            if (lhs.getNumRows() != rhs.getNumRows() ||
                lhs.getNumCols() != rhs.getNumCols() ||
                lhs.getNumRows() == -1 || lhs.getNumCols() == -1)
                return true;

            return false;
        }

        if (op->getOperandTypes()[0].isa<mlir::daphne::MatrixType>()) {
            mlir::daphne::MatrixType lhsMatrixType =
                op->getOperandTypes()[0].dyn_cast<mlir::daphne::MatrixType>();
            return lhsMatrixType.getNumRows() == -1 || lhsMatrixType.getNumCols() == -1;
        }

        return false;
    });

    populateLowerEwOpConversionPatterns(typeConverter, patterns);

    auto module = getOperation();
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
        signalPassFailure();
}

std::unique_ptr<mlir::Pass> mlir::daphne::createEwOpLoweringPass() {
    return std::make_unique<EwOpLoweringPass>();
}
