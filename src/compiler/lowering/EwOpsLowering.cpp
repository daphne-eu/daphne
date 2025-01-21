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

#include <memory>
#include <utility>

#include <compiler/utils/LoweringUtils.h>
#include <util/ErrorHandler.h>

#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

// ****************************************************************************
// Rewriter Templates (Elemwise Unary, Elemwise Binary)
// ****************************************************************************

using unaryFuncType = Value (*)(OpBuilder &rewriter, Location loc, TypeConverter *typeConverter, Value arg);

/**
 * @brief template for lowering elemwise unary functions.
 * The corresponding `UnaryOp` is applied to every element
 * of a matrix or scalar operand.
 *
 * @param UnaryOp The target operation this pass aims to rewrite.
 * @param UnaryFunc The function applied to every element. Must have
 * the following signature
 * `(OpBuilder, mlir::Location, TypeConverter*, mlir::Value arg) -> mlir::Value`.
 */
template <class UnaryOp, unaryFuncType unaryFunc> struct UnaryOpLowering : public mlir::OpConversionPattern<UnaryOp> {
  public:
    using OpAdaptor = typename mlir::OpConversionPattern<UnaryOp>::OpAdaptor;
    UnaryOpLowering(TypeConverter &typeConverter, mlir::MLIRContext *ctx)
        : mlir::OpConversionPattern<UnaryOp>(typeConverter, ctx) {
        this->setDebugName("EwDaphneOpsLowering");
    }

    LogicalResult matchAndRewriteScalarVal(UnaryOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
        rewriter.replaceOp(op, unaryFunc(rewriter, op->getLoc(), this->typeConverter, op.getArg()));

        return mlir::success();
    }

    LogicalResult matchAndRewrite(UnaryOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {

        Location loc = op->getLoc();
        daphne::MatrixType matrixType = adaptor.getArg().getType().template dyn_cast<daphne::MatrixType>();

        // Scalar values are handled separately. Otherwise assume input is DenseMatrix.
        if (!matrixType) {
            return matchAndRewriteScalarVal(op, adaptor, rewriter);
        }

        Type matrixElementType = matrixType.getElementType();
        ssize_t numRows = matrixType.getNumRows();
        ssize_t numCols = matrixType.getNumCols();

        if (numRows < 0 || numCols < 0) {
            throw ErrorHandler::compilerError(
                loc, "EwOpsLowering (BinaryOp)",
                "ewOps codegen currently only works with matrix dimensions that are known at compile time");
        }

        Value argMemref = rewriter.create<daphne::ConvertDenseMatrixToMemRef>(
            loc, MemRefType::get({numRows, numCols}, matrixElementType), adaptor.getArg());

        Value resMemref = rewriter.create<memref::AllocOp>(loc, MemRefType::get({numRows, numCols}, matrixElementType));

        SmallVector<AffineMap, 2> indexMaps = {AffineMap::getMultiDimIdentityMap(2, rewriter.getContext()),
                                               AffineMap::getMultiDimIdentityMap(2, rewriter.getContext())};
        SmallVector<utils::IteratorType, 2> iterTypes = {utils::IteratorType::parallel, utils::IteratorType::parallel};

        rewriter.create<linalg::GenericOp>(
            loc, TypeRange{}, ValueRange{argMemref}, ValueRange{resMemref}, indexMaps, iterTypes,
            [&](OpBuilder &OpBuilderNested, Location locNested, ValueRange arg) {
                Value resValue = unaryFunc(OpBuilderNested, locNested, this->typeConverter, arg[0]);
                OpBuilderNested.create<linalg::YieldOp>(locNested, resValue);
            });

        auto resDenseMatrix = convertMemRefToDenseMatrix(loc, rewriter, resMemref, op.getType());

        rewriter.replaceOp(op, resDenseMatrix);

        return mlir::success();
    }
};

using binaryFuncType = Value (*)(OpBuilder &rewriter, Location loc, TypeConverter *typeConverter, Value lhs, Value rhs);

/**
 * @brief template for lowering elemwise binary functions.
 * The corresponding `BinaryOp` is applied to every element
 * with the same index for 2 matching matrices, 2 scalar inputs,
 * or broadcasted to a matrix (lhs) from a scalar (rhs).
 *
 * @param BinaryOp The target operation this pass aims to rewrite.
 * @param BinaryFunc The function applied to every element pair. Must have
 * the following signature
 * `(OpBuilder, mlir::Location, TypeConverter*, mlir::Value lhs, mlir::Value rhs) -> mlir::Value`.
 */
template <class BinaryOp, binaryFuncType binaryFunc>
class BinaryOpLowering final : public mlir::OpConversionPattern<BinaryOp> {
  public:
    using OpAdaptor = typename mlir::OpConversionPattern<BinaryOp>::OpAdaptor;

    BinaryOpLowering(TypeConverter &typeConverter, mlir::MLIRContext *ctx)
        : mlir::OpConversionPattern<BinaryOp>(typeConverter, ctx) {
        this->setDebugName("EwDaphneOpLowering");
    }

    /**
     * @brief Returns an affine map for indexing the rhs operand.
     * Assumes that neither matrix is a singleton and lhs is not broadcast.
     *
     * If rhs has no dimensions of size 1, returns an identity map.
     * Else, returns a map (i,j)->(0,j) or (i,j)->(i,0) to enable broadcasting of rhs.
     */
    AffineMap buildRhsAffineMap(Location loc, ConversionPatternRewriter &rewriter, ssize_t lhsRows, ssize_t lhsCols,
                                ssize_t rhsRows, ssize_t rhsCols) const {

        AffineMap rhsAffineMap;

        // lhs could also be a row/column vector which should not be handled as broadcasting (even though the resulting
        // affine maps coincide). This allows for a clearer error message as well.
        if (lhsRows != 1 && rhsRows == 1) {
            // rhs is a row vector, broadcast along columns
            if (lhsCols != rhsCols) {
                throw ErrorHandler::compilerError(
                    loc, "EwOpsLowering (BinaryOp)",
                    "could not broadcast rhs along columns. Rhs must "
                    "be a scalar value, singleton matrix or have an equal amount of column to "
                    "be broadcast but operands have dimensions (" +
                        std::to_string(lhsRows) + "," + std::to_string(lhsCols) + ") and (" + std::to_string(rhsRows) +
                        "," + std::to_string(rhsCols) + ")");
            }
            rhsAffineMap = AffineMap::get(2, 0, {rewriter.getAffineConstantExpr(0), rewriter.getAffineDimExpr(1)},
                                          rewriter.getContext());
        } else if (lhsCols != 1 && rhsCols == 1) {
            // rhs is a column vector, broadcast along rows
            if (lhsRows != rhsRows) {
                throw ErrorHandler::compilerError(
                    loc, "EwOpsLowering (BinaryOp)",
                    "could not broadcast rhs along rows. Rhs must "
                    "be a scalar value, singleton matrix or have an equal amount of rows to "
                    "be broadcast but operands have dimensions (" +
                        std::to_string(lhsRows) + "," + std::to_string(lhsCols) + ") and (" + std::to_string(rhsRows) +
                        "," + std::to_string(rhsCols) + ")");
            }
            rhsAffineMap = AffineMap::get(2, 0, {rewriter.getAffineDimExpr(0), rewriter.getAffineConstantExpr(0)},
                                          rewriter.getContext());
        } else {
            // rhs is not broadcasted, return identity mapping
            if (lhsRows != rhsRows || lhsCols != rhsCols) {
                throw ErrorHandler::compilerError(
                    loc, "EwOpsLowering (BinaryOp)",
                    "lhs and rhs must have equal dimensions or allow for broadcasting but operands have dimensions (" +
                        std::to_string(lhsRows) + "," + std::to_string(lhsCols) + ") and (" + std::to_string(rhsRows) +
                        "," + std::to_string(rhsCols) + ")");
            }
            rhsAffineMap = AffineMap::getMultiDimIdentityMap(2, rewriter.getContext());
        }

        return rhsAffineMap;
    }

    LogicalResult matchAndRewriteScalarVal(BinaryOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
        rewriter.replaceOp(op,
                           binaryFunc(rewriter, op.getLoc(), this->typeConverter, adaptor.getLhs(), adaptor.getRhs()));
        return mlir::success();
    }

    LogicalResult matchAndRewriteBroadcastScalarRhs(BinaryOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter,
                                                    Value &rhs) const {
        Location loc = op->getLoc();
        Value lhs = adaptor.getLhs();

        auto lhsMatrixType = lhs.getType().template dyn_cast<daphne::MatrixType>();
        ssize_t lhsRows = lhsMatrixType.getNumRows();
        ssize_t lhsCols = lhsMatrixType.getNumCols();

        Type matrixElementType = lhsMatrixType.getElementType();

        MemRefType argMemRefType = MemRefType::get({lhsRows, lhsCols}, matrixElementType);
        auto lhsMemref = rewriter.create<daphne::ConvertDenseMatrixToMemRef>(loc, argMemRefType, lhs);

        Value resMemref = rewriter.create<memref::AllocOp>(loc, argMemRefType);

        SmallVector<AffineMap, 2> indexMaps = {AffineMap::getMultiDimIdentityMap(2, rewriter.getContext()),
                                               AffineMap::getMultiDimIdentityMap(2, rewriter.getContext())};
        SmallVector<utils::IteratorType, 2> iterTypes = {utils::IteratorType::parallel, utils::IteratorType::parallel};

        rewriter.create<linalg::GenericOp>(
            loc, TypeRange{}, ValueRange{lhsMemref}, ValueRange{resMemref}, indexMaps, iterTypes,
            [&](OpBuilder &OpBuilderNested, Location locNested, ValueRange arg) {
                Value resValue = binaryFunc(OpBuilderNested, locNested, this->typeConverter, arg[0], rhs);
                OpBuilderNested.create<linalg::YieldOp>(locNested, resValue);
            });

        Value resDenseMatrix = convertMemRefToDenseMatrix(loc, rewriter, resMemref, op.getType());

        rewriter.replaceOp(op, resDenseMatrix);
        return mlir::success();
    }

    LogicalResult matchAndRewrite(BinaryOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
        Location loc = op->getLoc();
        Value lhs = adaptor.getLhs();
        Value rhs = adaptor.getRhs();

        auto lhsMatrixType = lhs.getType().template dyn_cast<daphne::MatrixType>();
        auto rhsMatrixType = rhs.getType().template dyn_cast<daphne::MatrixType>();

        // Match Scalar-Scalar and Matrix-Scalar broadcasting (assuming scalar values are always switched to
        // rhs). Broadcasting where either Matrix is a singleton or vector needs to be handled separately below.
        if (!rhsMatrixType) {
            if (!lhsMatrixType) {
                return matchAndRewriteScalarVal(op, adaptor, rewriter);
            }
            return matchAndRewriteBroadcastScalarRhs(op, adaptor, rewriter, rhs);
        }

        Type matrixElementType = lhsMatrixType.getElementType();

        ssize_t lhsRows = lhsMatrixType.getNumRows();
        ssize_t lhsCols = lhsMatrixType.getNumCols();
        ssize_t rhsRows = rhsMatrixType.getNumRows();
        ssize_t rhsCols = rhsMatrixType.getNumCols();

        if (lhsRows < 0 || lhsCols < 0 || rhsRows < 0 || rhsCols < 0) {
            throw ErrorHandler::compilerError(
                loc, "EwOpsLowering (BinaryOp)",
                "ewOps codegen currently only works with matrix dimensions that are known at compile time");
        }

        // For efficiency, broadcasting a singleton is handled separately here (assumes singleton is always rhs).
        // Broadcasting of row/column vectors is handled during the construction of the index map for rhs below.
        if ((lhsRows != 1 || lhsCols != 1) && rhsRows == 1 && rhsCols == 1) {
            auto rhsMemref = rewriter.create<daphne::ConvertDenseMatrixToMemRef>(
                loc, MemRefType::get({1, 1}, matrixElementType), rhs);
            Value rhsBroadcastVal =
                rewriter
                    .create<memref::LoadOp>(loc, rhsMemref,
                                            ValueRange{rewriter.create<arith::ConstantIndexOp>(loc, 0),
                                                       rewriter.create<arith::ConstantIndexOp>(loc, 0)})
                    .getResult();
            return matchAndRewriteBroadcastScalarRhs(op, adaptor, rewriter, rhsBroadcastVal);
        }

        MemRefType lhsMemRefType = MemRefType::get({lhsRows, lhsCols}, matrixElementType);
        MemRefType rhsMemRefType = MemRefType::get({rhsRows, rhsCols}, matrixElementType);
        auto lhsMemref = rewriter.create<daphne::ConvertDenseMatrixToMemRef>(loc, lhsMemRefType, lhs);
        auto rhsMemref = rewriter.create<daphne::ConvertDenseMatrixToMemRef>(loc, rhsMemRefType, rhs);

        // If any broadcasting occurs, it is assumed to be rhs so res inherits its shape from lhs.
        Value resMemref = rewriter.create<memref::AllocOp>(loc, lhsMemRefType);

        // Builds an affine map to index the args and accounts for broadcasting of rhs.
        // Creation of rhs indexing map checks whether or not the dimensions match and returns a compiler error if not.
        SmallVector<AffineMap, 3> indexMaps = {AffineMap::getMultiDimIdentityMap(2, rewriter.getContext()),
                                               buildRhsAffineMap(loc, rewriter, lhsRows, lhsCols, rhsRows, rhsCols),
                                               AffineMap::getMultiDimIdentityMap(2, rewriter.getContext())};
        SmallVector<utils::IteratorType, 2> iterTypes = {utils::IteratorType::parallel, utils::IteratorType::parallel};

        rewriter.create<linalg::GenericOp>(
            loc, TypeRange{}, ValueRange{lhsMemref, rhsMemref}, ValueRange{resMemref}, indexMaps, iterTypes,
            [&](OpBuilder &OpBuilderNested, Location locNested, ValueRange arg) {
                Value resValue = binaryFunc(OpBuilderNested, locNested, this->typeConverter, arg[0], arg[1]);
                OpBuilderNested.create<linalg::YieldOp>(locNested, resValue);
            });

        Value resDenseMatrix = convertMemRefToDenseMatrix(loc, rewriter, resMemref, op.getType());

        rewriter.replaceOp(op, resDenseMatrix);
        return mlir::success();
    }
};

// ****************************************************************************
// Unary/Binary Function Specializations
// ****************************************************************************

// ----------------------------------------------------------------------------
// Generic Function Templates
// ----------------------------------------------------------------------------

template <typename IOp, typename FOp>
Value unaryNoConversionFunc(OpBuilder &rewriter, Location loc, TypeConverter *typeConverter, Value arg) {
    Value res = llvm::isa<mlir::IntegerType>(arg.getType()) ? rewriter.create<IOp>(loc, arg).getResult()
                                                            : rewriter.create<FOp>(loc, arg).getResult();
    return res;
}

template <typename IOp, typename FOp>
Value unaryWithConversionFunc(OpBuilder &rewriter, Location loc, TypeConverter *typeConverter, Value arg) {
    Type resType = arg.getType();
    Value res = arg;
    if (llvm::isa<mlir::IntegerType>(resType)) {
        res = convertToSignlessInt(rewriter, loc, typeConverter, res, resType);
        res = rewriter.create<IOp>(loc, res).getResult();
        res = typeConverter->materializeTargetConversion(rewriter, loc, resType, res);
    } else {
        res = rewriter.create<FOp>(loc, res).getResult();
    }
    return res;
}

template <typename IOp, typename FOp>
Value binaryWithConversionFunc(OpBuilder &rewriter, Location loc, TypeConverter *typeConverter, Value lhs, Value rhs) {
    Type resType = lhs.getType();
    Value res{};
    if (llvm::isa<mlir::IntegerType>(resType)) {
        lhs = convertToSignlessInt(rewriter, loc, typeConverter, lhs, resType);
        rhs = convertToSignlessInt(rewriter, loc, typeConverter, rhs, resType);
        res = rewriter.create<IOp>(loc, lhs, rhs).getResult();
        res = typeConverter->materializeTargetConversion(rewriter, loc, resType, res);
    } else {
        res = rewriter.create<FOp>(loc, lhs, rhs).getResult();
    }
    return res;
}

template <typename SIOp, typename UIOp, typename FOp>
Value binaryWithConversionFunc(OpBuilder &rewriter, Location loc, TypeConverter *typeConverter, Value lhs, Value rhs) {
    Type resType = lhs.getType();
    Value res{};
    if (llvm::isa<IntegerType>(resType)) {
        lhs = convertToSignlessInt(rewriter, loc, typeConverter, lhs, resType);
        rhs = convertToSignlessInt(rewriter, loc, typeConverter, rhs, resType);
        res = resType.isSignedInteger() ? rewriter.create<SIOp>(loc, lhs, rhs).getResult()
                                        : rewriter.create<UIOp>(loc, lhs, rhs).getResult();
        res = typeConverter->materializeTargetConversion(rewriter, loc, resType, res);
    } else {
        res = rewriter.create<FOp>(loc, lhs, rhs).getResult();
    }
    return res;
}

// ----------------------------------------------------------------------------
// Specialized Function Templates
// ----------------------------------------------------------------------------

// powOp has different specializations for certain combinations of value types
Value ewPowOpComputeRes(OpBuilder &rewriter, Location loc, TypeConverter *typeConverter, Value lhs, Value rhs) {
    Value resValue;
    Type rhsMatrixElementType = rhs.getType();
    Type resMatrixElementType = lhs.getType();
    // The integer specializations of PowOp expect signless Integers
    if (llvm::isa<mlir::IntegerType>(resMatrixElementType)) {
        Value lhsCasted = convertToSignlessInt(rewriter, loc, typeConverter, lhs, resMatrixElementType);
        Value rhsCasted = convertToSignlessInt(rewriter, loc, typeConverter, rhs, resMatrixElementType);
        resValue = rewriter.create<math::IPowIOp>(loc, lhsCasted, rhsCasted).getResult();
        resValue = typeConverter->materializeTargetConversion(rewriter, loc, resMatrixElementType, resValue);
    } else if (llvm::isa<mlir::IntegerType>(rhsMatrixElementType)) {
        Value rhsCasted = convertToSignlessInt(rewriter, loc, typeConverter, rhs, resMatrixElementType);
        resValue = rewriter.create<math::FPowIOp>(loc, lhs, rhsCasted).getResult();
    } else {
        resValue = rewriter.create<math::PowFOp>(loc, lhs, rhs).getResult();
    }
    return resValue;
}

// ****************************************************************************
// Rewriter Class Instantiations
// ****************************************************************************

// Unary Arithmetic/general math
using AbsOpLowering = UnaryOpLowering<daphne::EwAbsOp, unaryWithConversionFunc<math::AbsIOp, math::AbsFOp>>;
// DAPHNE promotes argument type of sqrt to f32/64, so SqrtOp does not deal with integer values
using SqrtOpLowering = UnaryOpLowering<daphne::EwSqrtOp, unaryNoConversionFunc<math::SqrtOp, math::SqrtOp>>;
using ExpOpLowering = UnaryOpLowering<daphne::EwExpOp, unaryNoConversionFunc<math::ExpOp, math::ExpOp>>;
using LnOpLowering = UnaryOpLowering<daphne::EwLnOp, unaryNoConversionFunc<math::LogOp, math::LogOp>>;

// Unary Trig/Hyperbolic functions
using SinOpLowering = UnaryOpLowering<daphne::EwSinOp, unaryNoConversionFunc<math::SinOp, math::SinOp>>;
using CosOpLowering = UnaryOpLowering<daphne::EwCosOp, unaryNoConversionFunc<math::CosOp, math::CosOp>>;
// TODO: link needed library for other trigonometric operations
// using TanOpLowering = UnaryOpLowering<daphne::EwTanOp, unaryNoConversionFunc<math::TanOp, math::TanOp>>;
// using AsinOpLowering = UnaryOpLowering<daphne::EwAsinOp, unaryNoConversionFunc<math::AsinOp, math::AsinOp>>;
// using AcosOpLowering = UnaryOpLowering<daphne::EwAcosOp, unaryNoConversionFunc<math::AcosOp, math::AcosOp>>;
// using AtanOpLowering = UnaryOpLowering<daphne::EwAtanOp, unaryNoConversionFunc<math::AtanOp, math::AtanOp>>;
// using SinhOpLowering = UnaryOpLowering<daphne::EwSinhOp, unaryNoConversionFunc<math::SinhOp, math::SinhOp>>;
// using CoshOpLowering = UnaryOpLowering<daphne::EwCoshOp, unaryNoConversionFunc<math::CoshOp, math::CoshOp>>;
// using TanhOpLowering = UnaryOpLowering<daphne::EwTanhOp, unaryNoConversionFunc<math::TanhOp, math::TanhOp>>;

// Rounding
// Prior canonicalization pass removes rounding ops on integers, meaning only f32/f64 types need to be dealt
// with
using FloorOpLowering = UnaryOpLowering<daphne::EwFloorOp, unaryNoConversionFunc<math::FloorOp, math::FloorOp>>;
using CeilOpLowering = UnaryOpLowering<daphne::EwCeilOp, unaryNoConversionFunc<math::CeilOp, math::CeilOp>>;
using RoundOpLowering = UnaryOpLowering<daphne::EwRoundOp, unaryNoConversionFunc<math::RoundOp, math::RoundOp>>;

// Binary Arithmetic/general math
using AddOpLowering = BinaryOpLowering<daphne::EwAddOp, binaryWithConversionFunc<arith::AddIOp, arith::AddFOp>>;
using SubOpLowering = BinaryOpLowering<daphne::EwSubOp, binaryWithConversionFunc<arith::SubIOp, arith::SubFOp>>;
using MulOpLowering = BinaryOpLowering<daphne::EwMulOp, binaryWithConversionFunc<arith::MulIOp, arith::MulFOp>>;
using DivOpLowering =
    BinaryOpLowering<daphne::EwDivOp, binaryWithConversionFunc<arith::DivSIOp, arith::DivUIOp, arith::DivFOp>>;
// using PowOpLowering = BinaryOpLowering<daphne::EwPowOp, ewPowOpComputeRes>; // TODO: link needed library
// ModOpLowering - specialized in ModOpLowering.cpp
// TODO: find or implement generalized logarithm in mlir

// Binary Comparison
// Min/Max
using MaxOpLowering =
    BinaryOpLowering<daphne::EwMinOp, binaryWithConversionFunc<arith::MinSIOp, arith::MinUIOp, arith::MinFOp>>;
using MinOpLowering =
    BinaryOpLowering<daphne::EwMaxOp, binaryWithConversionFunc<arith::MaxSIOp, arith::MaxUIOp, arith::MaxFOp>>;

// Logical
// using AndOpLowering =
//     BinaryOpLowering<daphne::EwAndOp, binaryWithConversionFunc<arith::AndIOp, arith::AndIOp>>; // distinguish
//     AndFOp
// using OrOpLowering = BinaryOpLowering<daphne::EwOrOp, binaryWithConversionFunc<arith::OrIOp, arith::OrIOp>>;
// // - " -

// ****************************************************************************
// General Pass Setup
// ****************************************************************************

namespace {
/**
 * @brief This pass lowers element-wise operations to Linalg GenericOps
 * and arithmetic operations.
 *
 * This rewrite may enable loop fusion of the produced affine loops by
 * running the loop fusion pass.
 */
struct EwOpLoweringPass : public mlir::PassWrapper<EwOpLoweringPass, mlir::OperationPass<mlir::ModuleOp>> {
    explicit EwOpLoweringPass() = default;

    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry
            .insert<mlir::LLVM::LLVMDialect, mlir::AffineDialect, memref::MemRefDialect, mlir::linalg::LinalgDialect,
                    daphne::DaphneDialect, mlir::math::MathDialect, mlir::arith::ArithDialect>();
    }
    void runOnOperation() final;

    [[nodiscard]] StringRef getArgument() const final { return "lower-ew"; }
    [[nodiscard]] StringRef getDescription() const final {
        return "This pass lowers element-wise operations to Linalg GenericOps "
               "that lower to affine loops and arithmetic operations.";
    }
};
} // end anonymous namespace

void populateLowerEwOpConversionPatterns(mlir::LLVMTypeConverter &typeConverter, mlir::RewritePatternSet &patterns) {
    // clang-format off
    patterns.insert<
        // UnaryOps
        AbsOpLowering,
        SqrtOpLowering,
        ExpOpLowering,
        LnOpLowering,
        SinOpLowering,
        CosOpLowering,
        // TanOpLowering,
        // AsinOpLowering,
        // AcosOpLowering,
        // AtanOpLowering,
        // TanhOpLowering,
        // CoshOpLowering,
        // TanhOpLowering,
        FloorOpLowering,
        CeilOpLowering,
        RoundOpLowering,
        // BinaryOps
        AddOpLowering,
        SubOpLowering,
        MulOpLowering,
        DivOpLowering,
        // PowOpLowering,
        MinOpLowering,
        MaxOpLowering
        // , AndOpLowering,
        // OrOpLowering
        >(typeConverter, patterns.getContext());
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

    target.addLegalDialect<mlir::arith::ArithDialect, memref::MemRefDialect, mlir::AffineDialect,
                           mlir::LLVM::LLVMDialect, daphne::DaphneDialect, mlir::BuiltinDialect,
                           mlir::math::MathDialect, mlir::linalg::LinalgDialect>();

    // UnaryOps
    target.addDynamicallyLegalOp<daphne::EwAbsOp, daphne::EwSqrtOp, daphne::EwExpOp, daphne::EwLnOp, daphne::EwSinOp,
                                 daphne::EwCosOp,
                                 /* daphne::EwTanOp, daphne::EwAsinOp, daphne::EwAcosOp, daphne::EwAtanOp,
                                  daphne::EwSinhOp, daphne::EwCoshOp, daphne::EwTanhOp,*/
                                 daphne::EwFloorOp, daphne::EwCeilOp, daphne::EwRoundOp>([](Operation *op) {
        Type operand = op->getOperand(0).getType();
        if (llvm::isa<IntegerType>(operand) || llvm::isa<FloatType>(operand)) {
            return false;
        }
        auto matType = operand.dyn_cast<daphne::MatrixType>();
        if (matType && matType.getRepresentation() == daphne::MatrixRepresentation::Dense) {
            return false;
        }
        return true;
    });

    // BinaryOps
    target
        .addDynamicallyLegalOp<daphne::EwAddOp, daphne::EwSubOp, daphne::EwMulOp, daphne::EwDivOp, /*daphne::EwPowOp,*/
                               daphne::EwMinOp, daphne::EwMaxOp /*, daphne::EwAndOp, daphne::EwOrOp*/>(
            [](Operation *op) {
                Type lhs = op->getOperand(0).getType();
                Type rhs = op->getOperand(1).getType();
                auto lhsMatType = lhs.dyn_cast<daphne::MatrixType>();
                auto rhsMatType = rhs.dyn_cast<daphne::MatrixType>();
                // Rhs is scalar and lhs is scalar or dense matrix (rhs is broadcasted)
                if ((llvm::isa<IntegerType>(rhs) || llvm::isa<FloatType>(rhs)) &&
                    ((llvm::isa<IntegerType>(lhs) || llvm::isa<FloatType>(lhs)) ||
                     (lhsMatType && lhsMatType.getRepresentation() == daphne::MatrixRepresentation::Dense))) {
                    return false;
                }
                // Both sides are dense matrices (rhs might still be broadcasted if it is a singleton)
                if ((lhsMatType && lhsMatType.getRepresentation() == daphne::MatrixRepresentation::Dense) &&
                    (rhsMatType && rhsMatType.getRepresentation() == daphne::MatrixRepresentation::Dense)) {
                    return false;
                }
                return true;
            });

    populateLowerEwOpConversionPatterns(typeConverter, patterns);

    auto module = getOperation();
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
        signalPassFailure();
}

std::unique_ptr<mlir::Pass> daphne::createEwOpLoweringPass() { return std::make_unique<EwOpLoweringPass>(); }
