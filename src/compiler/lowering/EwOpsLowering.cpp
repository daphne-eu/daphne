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
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "util/ErrorHandler.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

template <class UnaryOp, class IOp, class FOp>
struct UnaryOpLowering : public mlir::OpConversionPattern<UnaryOp> {
public:
    using OpAdaptor = typename mlir::OpConversionPattern<UnaryOp>::OpAdaptor;
    UnaryOpLowering(mlir::TypeConverter &typeConverter, mlir::MLIRContext *ctx)
        : mlir::OpConversionPattern<UnaryOp>(typeConverter, ctx) {
        this->setDebugName("EwDaphneOpsLowering");
    }

    mlir::LogicalResult matchAndRewriteScalarVal(
        UnaryOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const {

        mlir::Value arg = op.getArg();
        mlir::Type argType = arg.getType();

        if (llvm::isa<mlir::IntegerType>(argType))
            rewriter.replaceOpWithNewOp<IOp>(op.getOperation(), arg);
        else if (llvm::isa<mlir::FloatType>(argType))
            rewriter.replaceOpWithNewOp<FOp>(op.getOperation(), arg);

        return mlir::success();
    }


    mlir::LogicalResult matchAndRewrite(
        UnaryOp op, OpAdaptor adaptor,
        mlir::ConversionPatternRewriter &rewriter) const override {

        mlir::Location loc = op->getLoc();
        mlir::daphne::MatrixType matrixType = adaptor.getArg().getType().template dyn_cast<mlir::daphne::MatrixType>();

        // Scalar values are handled separately. The rest of the code assumes the input to be dense matrices.
        if (!matrixType) {
            return matchAndRewriteScalarVal(op, adaptor, rewriter);
        }

        mlir::Type matrixElementType = matrixType.getElementType();
        ssize_t numRows = matrixType.getNumRows();
        ssize_t numCols = matrixType.getNumCols();

        mlir::Value argMemref = rewriter.create<mlir::daphne::ConvertDenseMatrixToMemRef>(loc,
            mlir::MemRefType::get({numRows, numCols}, matrixElementType),
            adaptor.getArg()
        );

        Value resMemref = rewriter.create<mlir::memref::AllocOp>(loc,
            mlir::MemRefType::get({numRows, numCols}, matrixElementType)
        );

        SmallVector<AffineMap, 2> indexMaps = {
            AffineMap::getMultiDimIdentityMap(2, rewriter.getContext()),
            AffineMap::getMultiDimIdentityMap(2, rewriter.getContext())
        };
        SmallVector<utils::IteratorType, 2> iterTypes = {
            utils::IteratorType::parallel,
            utils::IteratorType::parallel
        };

        rewriter.create<linalg::GenericOp>(loc,
            /*Output Type*/ TypeRange{},
            /*Inputs*/  ValueRange{argMemref},
            /*Outputs*/ ValueRange{resMemref},
            indexMaps,
            iterTypes,
            /*GenericOp body*/
            [&](OpBuilder &OpBuilderNested, Location locNested, ValueRange arg){
                auto mappedArg = llvm::isa<mlir::IntegerType>(matrixElementType)
                        ? OpBuilderNested.create<IOp>(locNested, arg[0]).getResult()
                        : OpBuilderNested.create<FOp>(locNested, arg[0]).getResult();

                OpBuilderNested.create<linalg::YieldOp>(locNested, mappedArg);
            }
        );

        auto resDenseMatrix = convertMemRefToDenseMatrix(loc, rewriter, resMemref, op.getType());

        rewriter.replaceOp(op, resDenseMatrix);

        return mlir::success();
    }
};

template <class BinaryOp, class IOp, class FOp>
class BinaryOpLowering final : public mlir::OpConversionPattern<BinaryOp> {
public:
    using OpAdaptor = typename mlir::OpConversionPattern<BinaryOp>::OpAdaptor;

  public:
    BinaryOpLowering(mlir::TypeConverter &typeConverter, mlir::MLIRContext *ctx)
        : mlir::OpConversionPattern<BinaryOp>(typeConverter, ctx) {
        this->setDebugName("EwDaphneOpLowering");
    }

    mlir::LogicalResult convertEwScalar(BinaryOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const {
        auto lhs = adaptor.getLhs();
        auto rhs = adaptor.getRhs();
        auto loc = op.getLoc();

        if (lhs.getType().template isa<mlir::FloatType>() && rhs.getType().template isa<mlir::FloatType>()) {
            rewriter.replaceOpWithNewOp<FOp>(op.getOperation(), adaptor.getOperands());
            return mlir::success();
        }

        Value castedLhs = this->typeConverter->materializeTargetConversion(
            rewriter, loc, rewriter.getIntegerType(adaptor.getRhs().getType().getIntOrFloatBitWidth()),
            ValueRange{adaptor.getLhs()});

        Value castedRhs = this->typeConverter->materializeTargetConversion(
            rewriter, loc, rewriter.getIntegerType(adaptor.getRhs().getType().getIntOrFloatBitWidth()),
            ValueRange{adaptor.getRhs()});

        Value binaryOp = rewriter.create<IOp>(loc, castedLhs, castedRhs);

        Value res =
            this->typeConverter->materializeSourceConversion(rewriter, loc, lhs.getType(), ValueRange{binaryOp});

        rewriter.replaceOp(op, res);
        return mlir::success();
    }

    mlir::LogicalResult matchAndRewrite(BinaryOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        auto lhs = adaptor.getLhs();
        auto rhs = adaptor.getRhs();

        // no matrix
        if (!lhs.getType().template isa<mlir::daphne::MatrixType>() &&
            !rhs.getType().template isa<mlir::daphne::MatrixType>())
            return convertEwScalar(op, adaptor, rewriter);

        // for now assume matrix is LHS and RHS is non matrix
        mlir::daphne::MatrixType lhsMatrixType =
            adaptor.getLhs().getType().template dyn_cast<mlir::daphne::MatrixType>();
        auto matrixElementType = lhsMatrixType.getElementType();
        auto lhsRows = lhsMatrixType.getNumRows();
        auto lhsCols = lhsMatrixType.getNumCols();
        auto lhsMemRefType = mlir::MemRefType::get({lhsRows, lhsCols}, matrixElementType);

        mlir::Type elementType{};
        mlir::Value memRefLhs =
            rewriter.create<mlir::daphne::ConvertDenseMatrixToMemRef>(op->getLoc(), lhsMemRefType, adaptor.getLhs());

        mlir::Value memRefRhs{};
        bool isMatrixMatrix = rhs.getType().template isa<mlir::daphne::MatrixType>();

        if (isMatrixMatrix) {
            memRefRhs = rewriter.create<mlir::daphne::ConvertDenseMatrixToMemRef>(op->getLoc(), lhsMemRefType,
                                                                                  adaptor.getRhs());
            elementType = lhsMemRefType.getElementType();
        } else {
            elementType = rhs.getType();
        }

        mlir::Value outputMemRef = insertMemRefAlloc(lhsMemRefType, op->getLoc(), rewriter);

        SmallVector<int64_t, 4> lowerBounds(/*Rank=*/2, /*Value=*/0);
        SmallVector<int64_t, 4> steps(/*Rank=*/2, /*Value=*/1);
        buildAffineLoopNest(
            rewriter, op.getLoc(), lowerBounds, {lhsMatrixType.getNumRows(), lhsMatrixType.getNumCols()}, steps,
            [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
                mlir::Value loadLhs = nestedBuilder.create<AffineLoadOp>(loc, memRefLhs, ivs);
                mlir::Value binaryOp{};

                if (adaptor.getRhs().getType().template isa<mlir::FloatType>()) {
                    binaryOp = nestedBuilder.create<FOp>(loc, loadLhs, adaptor.getRhs());

                    nestedBuilder.create<AffineStoreOp>(loc, binaryOp, outputMemRef, ivs);
                    return;
                }

                mlir::Value rhs{};
                if (isMatrixMatrix)
                    rhs = nestedBuilder.create<AffineLoadOp>(loc, memRefRhs, ivs);
                else
                    rhs = adaptor.getRhs();

                // is integer
                if (elementType.isInteger(elementType.getIntOrFloatBitWidth())) {
                    Value castedLhs = this->typeConverter->materializeTargetConversion(
                        nestedBuilder, loc, nestedBuilder.getIntegerType(lhsMemRefType.getElementTypeBitWidth()),
                        ValueRange{loadLhs});

                    Value castedRhs = this->typeConverter->materializeTargetConversion(
                        nestedBuilder, loc, nestedBuilder.getIntegerType(lhsMemRefType.getElementTypeBitWidth()),
                        ValueRange{rhs});

                    binaryOp = nestedBuilder.create<IOp>(loc, castedLhs, castedRhs);
                    Value castedRes = this->typeConverter->materializeSourceConversion(nestedBuilder, loc, elementType,
                                                                                       ValueRange{binaryOp});
                    nestedBuilder.create<AffineStoreOp>(loc, castedRes, outputMemRef, ivs);
                } else {
                    // is float
                    binaryOp = nestedBuilder.create<FOp>(loc, loadLhs, rhs);
                    nestedBuilder.create<AffineStoreOp>(loc, binaryOp, outputMemRef, ivs);
                }
            });
        mlir::Value output = convertMemRefToDenseMatrix(op->getLoc(), rewriter, outputMemRef, op.getType());

        rewriter.replaceOp(op, output);
        return mlir::success();
    }
};

// ****************************************************************************
// Generic EwOps that do not require type conversion inside
// ****************************************************************************

// clang-format off
// math::sqrt only supports floating point, DAPHNE promotes argument type of sqrt to f32/64
// Unary Arithmetic/general math
// AbsOpLowering - specialized below
using SqrtOpLowering    = UnaryOpLowering<daphne::EwSqrtOp,     math::SqrtOp,       math::SqrtOp>;
using ExpOpLowering     = UnaryOpLowering<daphne::EwExpOp,      math::ExpOp,        math::ExpOp>;
using LnOpLowering      = UnaryOpLowering<daphne::EwLnOp,       math::LogOp,        math::LogOp>;
// Unary Trig/Hyperbolic functions
using SinOpLowering     = UnaryOpLowering<daphne::EwSinOp,      math::SinOp,        math::SinOp>;
using CosOpLowering     = UnaryOpLowering<daphne::EwCosOp,      math::CosOp,        math::CosOp>;
// using TanOpLowering     = UnaryOpLowering<daphne::EwTanOp,      math::TanOp,        math::TanOp>;
// using AsinOpLowering    = UnaryOpLowering<daphne::EwAsinOp,     math::AsinOp,       math::AsinOp>;
// using AcosOpLowering    = UnaryOpLowering<daphne::EwAcosOp,     math::AcosOp,       math::AcosOp>;
// using AtanOpLowering    = UnaryOpLowering<daphne::EwAtanOp,     math::AtanOp,       math::AtanOp>;
// using SinhOpLowering    = UnaryOpLowering<daphne::EwSinhOp,     math::SinhOp,       math::SinhOp>;
// using CoshOpLowering    = UnaryOpLowering<daphne::EwCoshOp,     math::CoshOp,       math::CoshOp>;
// using TanhOpLowering    = UnaryOpLowering<daphne::EwTanhOp,     math::TanhOp,       math::TanhOp>;
/**
 * Rounding
 *
 * Since these operations have no effect on un/signed integers, they are removed in a prior canoniclization pass.
 * Therefore, the lowering for them only has to deal with floating point values.
 */
using FloorOpLowering   = UnaryOpLowering<daphne::EwFloorOp,    math::FloorOp,      math::FloorOp>;
using CeilOpLowering    = UnaryOpLowering<daphne::EwCeilOp,     math::CeilOp,       math::CeilOp>;
using RoundOpLowering   = UnaryOpLowering<daphne::EwRoundOp,    math::RoundOp,      math::RoundOp>;

// Binary Arithmetic/general math
using AddOpLowering     = BinaryOpLowering<daphne::EwAddOp,     arith::AddIOp,      arith::AddFOp>;
using SubOpLowering     = BinaryOpLowering<daphne::EwSubOp,     arith::SubIOp,      arith::SubFOp>;
using MulOpLowering     = BinaryOpLowering<daphne::EwMulOp,     arith::MulIOp,      arith::MulFOp>;
using DivOpLowering     = BinaryOpLowering<daphne::EwDivOp,     arith::DivSIOp,     arith::DivFOp>; // distinguish SI/UI
// PowOpLowering  - specialized below
// ModOpLowering - specialized in ModOpLowering.cpp
// TODO: find or implement generalized logarithm in mlir
// Binary Comparison
// Min/Max
using MinOpLowering     = BinaryOpLowering<daphne::EwMinOp,     arith::MinSIOp,     arith::MinFOp>; // distinguish Min SI/UI
using MaxOpLowering     = BinaryOpLowering<daphne::EwMaxOp,     arith::MaxSIOp,     arith::MaxFOp>; // - " -
// Logical
using AndOpLowering     = BinaryOpLowering<daphne::EwAndOp,     arith::AndIOp,      arith::AndIOp>; // distinguish AndFOp
using OrOpLowering      = BinaryOpLowering<daphne::EwOrOp,      arith::OrIOp,       arith::OrIOp>;  // - " -
// clang-format on


// ****************************************************************************
// UnaryOp specializations 
// ****************************************************************************

// ----------------------------------------------------------------------------
// Absolute value (AbsOp)
// ----------------------------------------------------------------------------

struct AbsOpLowering : public mlir::OpConversionPattern<daphne::EwAbsOp> {
public:
    using OpAdaptor = typename mlir::OpConversionPattern<daphne::EwAbsOp>::OpAdaptor;

    explicit AbsOpLowering(mlir::TypeConverter &typeConverter, mlir::MLIRContext *ctx)
            : mlir::OpConversionPattern<daphne::EwAbsOp>(typeConverter, ctx) {
        this->setDebugName("EwDaphneOpsLowering");
    }

    mlir::LogicalResult matchAndRewriteScalarVal(
        daphne::EwAbsOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const {

        mlir::Location loc = op.getLoc();
        mlir::Value mappedArg = op.getArg();
        mlir::Type resType = mappedArg.getType();

        if (llvm::isa<mlir::IntegerType>(resType)) {
            // math::AbsIOp expects a signless Integer
            mappedArg = this->typeConverter->materializeTargetConversion(rewriter, loc,
                rewriter.getIntegerType(resType.getIntOrFloatBitWidth()),
                ValueRange{mappedArg}
            );
            mappedArg = rewriter.create<math::AbsIOp>(loc, mappedArg).getResult();
            mappedArg = this->typeConverter->materializeTargetConversion(rewriter, loc,
                resType,
                ValueRange{mappedArg}
            );
        }
        else if (llvm::isa<mlir::FloatType>(resType)) {
            mappedArg = rewriter.create<math::AbsFOp>(loc, mappedArg).getResult();
        }
        rewriter.replaceOp(op, mappedArg);

        return mlir::success();
    }

    mlir::LogicalResult
    matchAndRewrite(daphne::EwAbsOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const override {

        mlir::Location loc = op->getLoc();
        mlir::daphne::MatrixType matrixType = adaptor.getArg().getType().template dyn_cast<mlir::daphne::MatrixType>();

        // Scalar values are handled separately. The rest of the code assumes the input to be dense matrices.
        if (!matrixType) {
            return matchAndRewriteScalarVal(op, adaptor, rewriter);
        }

        mlir::Type matrixElementType = matrixType.getElementType();
        ssize_t numRows = matrixType.getNumRows();
        ssize_t numCols = matrixType.getNumCols();

        mlir::Value argMemref = rewriter.create<mlir::daphne::ConvertDenseMatrixToMemRef>(loc,
            mlir::MemRefType::get({numRows, numCols}, matrixElementType),
            adaptor.getArg()
        );

        Value resMemref = rewriter.create<mlir::memref::AllocOp>(loc,
            mlir::MemRefType::get({numRows, numCols}, matrixElementType)
        );

        SmallVector<AffineMap, 2> indexMaps = {
            AffineMap::getMultiDimIdentityMap(2, rewriter.getContext()),
            AffineMap::getMultiDimIdentityMap(2, rewriter.getContext())
        };
        SmallVector<utils::IteratorType, 2> iterTypes = {utils::IteratorType::parallel, utils::IteratorType::parallel};
        rewriter.create<linalg::GenericOp>(loc,
            TypeRange{},
            ValueRange{argMemref},
            ValueRange{resMemref},
            indexMaps,
            iterTypes,
            [&](OpBuilder &OpBuilderNested, Location locNested, ValueRange arg){
                mlir::Value mappedArg = arg[0];

                if (llvm::isa<mlir::IntegerType>(matrixElementType)) {
                    // math::AbsIOp expects a signless Integer
                    mappedArg = this->typeConverter->materializeTargetConversion(rewriter, loc,
                        rewriter.getIntegerType(matrixElementType.getIntOrFloatBitWidth()),
                        ValueRange{mappedArg}
                    );
                    mappedArg = OpBuilderNested.create<math::AbsIOp>(locNested, mappedArg).getResult();
                    mappedArg = this->typeConverter->materializeTargetConversion(rewriter, loc,
                        matrixElementType,
                        ValueRange{mappedArg}
                    );
                }
                else if (llvm::isa<mlir::FloatType>(matrixElementType)) {
                    mappedArg = OpBuilderNested.create<math::AbsFOp>(locNested, mappedArg).getResult();
                }

                rewriter.create<linalg::YieldOp>(locNested, mappedArg);
            }
        );
        auto resDenseMatrix = convertMemRefToDenseMatrix(loc, rewriter, resMemref, op.getType());
        rewriter.replaceOp(op, resDenseMatrix);

        return mlir::success();
    }
};

// ****************************************************************************
// BinaryOp specializations 
// ****************************************************************************

// ----------------------------------------------------------------------------
// Exponentiation (PowOp)
// ----------------------------------------------------------------------------

struct PowOpLowering : public mlir::OpConversionPattern<daphne::EwPowOp> {
public:
    using OpAdaptor = typename mlir::OpConversionPattern<daphne::EwPowOp>::OpAdaptor;

    explicit PowOpLowering(mlir::TypeConverter &typeConverter, mlir::MLIRContext *ctx)
            : mlir::OpConversionPattern<daphne::EwPowOp>(typeConverter, ctx) {
        this->setDebugName("EwDaphneOpsLowering");
    }

    mlir::LogicalResult
    matchAndRewrite(daphne::EwPowOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const override {

        mlir::Location loc = op->getLoc();
        
        mlir::daphne::MatrixType lhsMatrixType = adaptor.getLhs().getType().template dyn_cast<mlir::daphne::MatrixType>();
        mlir::daphne::MatrixType rhsMatrixType = adaptor.getRhs().getType().template dyn_cast<mlir::daphne::MatrixType>();
        mlir::Type lhsMatrixElementType = lhsMatrixType.getElementType();
        mlir::Type rhsMatrixElementType = rhsMatrixType.getElementType();
        ssize_t numRows = lhsMatrixType.getNumRows();
        ssize_t numCols = lhsMatrixType.getNumCols();

        if (numRows != rhsMatrixType.getNumRows() || numCols != lhsMatrixType.getNumCols())
            throw ErrorHandler::compilerError(loc, "EwOpsLowering", "lhs and rhs do not have matching dimensions");

        mlir::Type resMatrixElementType;
        if (!llvm::isa<mlir::IntegerType>(lhsMatrixElementType)) {
            resMatrixElementType = lhsMatrixElementType;
        }
        else if (!llvm::isa<mlir::IntegerType>(rhsMatrixElementType)) {
            resMatrixElementType = rhsMatrixElementType;
        }
        else {
            resMatrixElementType = lhsMatrixElementType;
        }

        mlir::Value lhsMemref = rewriter.create<mlir::daphne::ConvertDenseMatrixToMemRef>(loc,
            mlir::MemRefType::get({numRows, numCols}, resMatrixElementType),
            adaptor.getLhs()
        );
        mlir::Value rhsMemref = rewriter.create<mlir::daphne::ConvertDenseMatrixToMemRef>(loc,
            mlir::MemRefType::get({numRows, numCols}, resMatrixElementType),
            adaptor.getRhs()
        );
        Value resMemref = rewriter.create<mlir::memref::AllocOp>(loc,
            mlir::MemRefType::get({numRows, numCols}, resMatrixElementType)
        );

        SmallVector<AffineMap, 3> indexMaps = {
            AffineMap::getMultiDimIdentityMap(2, rewriter.getContext()),
            AffineMap::getMultiDimIdentityMap(2, rewriter.getContext()),
            AffineMap::getMultiDimIdentityMap(2, rewriter.getContext())
        };
        SmallVector<utils::IteratorType, 2> iterTypes = {utils::IteratorType::parallel, utils::IteratorType::parallel};
        rewriter.create<linalg::GenericOp>(loc,
            TypeRange{},
            ValueRange{lhsMemref, rhsMemref},
            ValueRange{resMemref},
            indexMaps,
            iterTypes,
            [&](OpBuilder &OpBuilderNested, Location locNested, ValueRange arg){
                mlir::Value resValue;
                // The integer specializations of PowOp expect signless Integers
                if (llvm::isa<mlir::IntegerType>(resMatrixElementType)) {
                    mlir::Value lhsCasted = this->typeConverter->materializeTargetConversion(rewriter, loc,
                        rewriter.getIntegerType(resMatrixElementType.getIntOrFloatBitWidth()),
                        ValueRange{arg[0]}
                    );
                    mlir::Value rhsCasted = this->typeConverter->materializeTargetConversion(rewriter, loc,
                        rewriter.getIntegerType(resMatrixElementType.getIntOrFloatBitWidth()),
                        ValueRange{arg[1]}
                    );
                    resValue = OpBuilderNested.create<math::IPowIOp>(locNested, lhsCasted, rhsCasted).getResult();
                    resValue = this->typeConverter->materializeTargetConversion(rewriter, loc,
                        resMatrixElementType,
                        ValueRange{resValue}
                    );
                }
                else if (llvm::isa<mlir::IntegerType>(rhsMatrixElementType)) {
                    mlir::Value rhsCasted = this->typeConverter->materializeTargetConversion(rewriter, loc,
                        rewriter.getIntegerType(resMatrixElementType.getIntOrFloatBitWidth()),
                        ValueRange{arg[1]}
                    );
                    resValue = OpBuilderNested.create<math::FPowIOp>(locNested, arg[0], rhsCasted).getResult();
                }
                else {
                    resValue = OpBuilderNested.create<math::PowFOp>(locNested, arg[0], arg[1]).getResult();
                }

                rewriter.create<linalg::YieldOp>(locNested, resValue);
            }
        );

        auto resDenseMatrix = convertMemRefToDenseMatrix(loc, rewriter, resMemref, op.getType());
        rewriter.replaceOp(op, resDenseMatrix);

        return mlir::success();
    }
};

namespace {
/**
 * @brief This pass lowers element-wise operations to affine loop
 * structures and arithmetic operations.
 *
 * This rewrite may enable loop fusion of the produced affine loops by
 * running the loop fusion pass.
 */
struct EwOpLoweringPass : public mlir::PassWrapper<EwOpLoweringPass, mlir::OperationPass<mlir::ModuleOp>> {
    explicit EwOpLoweringPass() {}

    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<mlir::LLVM::LLVMDialect, mlir::AffineDialect,
                        mlir::memref::MemRefDialect, mlir::linalg::LinalgDialect,
                        mlir::daphne::DaphneDialect, mlir::math::MathDialect,
                        mlir::arith::ArithDialect>();
    }
    void runOnOperation() final;

    StringRef getArgument() const final { return "lower-ew"; }
    StringRef getDescription() const final {
        return "This pass lowers element-wise operations to affine-loop "
               "structures and arithmetic operations.";
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
                           mlir::BuiltinDialect, mlir::math::MathDialect,
                           mlir::linalg::LinalgDialect>();

    // target.addDynamicallyLegalOp<mlir::daphne::EwSqrtOp, mlir::daphne::EwAbsOp>(
    //     [](Operation *op) {
    //         return llvm::isa<mlir::daphne::MatrixType>(op->getOperandTypes()[0]);
    //     });

    // target.addDynamicallyLegalOp<mlir::daphne::EwAddOp, mlir::daphne::EwSubOp,
    //                              mlir::daphne::EwMulOp, mlir::daphne::EwPowOp,
    //                              mlir::daphne::EwDivOp>([](Operation *op) {
    //     if (llvm::isa<mlir::daphne::MatrixType>(op->getOperandTypes()[0]) &&
    //         llvm::isa<mlir::daphne::MatrixType>(op->getOperandTypes()[1])) {
    //         mlir::daphne::MatrixType lhs =
    //             op->getOperandTypes()[0]
    //                 .template dyn_cast<mlir::daphne::MatrixType>();
    //         mlir::daphne::MatrixType rhs =
    //             op->getOperandTypes()[1]
    //                 .template dyn_cast<mlir::daphne::MatrixType>();
    //         if (lhs.getNumRows() != rhs.getNumRows() ||
    //             lhs.getNumCols() != rhs.getNumCols() ||
    //             lhs.getNumRows() == -1 || lhs.getNumCols() == -1)
    //             return true;

    //         return false;
    //     }

    //     if (llvm::isa<mlir::daphne::MatrixType>(op->getOperandTypes()[0])) {
    //         mlir::daphne::MatrixType lhsMatrixType =
    //             op->getOperandTypes()[0].dyn_cast<mlir::daphne::MatrixType>();
    //         return lhsMatrixType.getNumRows() == -1 || lhsMatrixType.getNumCols() == -1;
    //     }

    //     return false;
    // });

    target.addIllegalOp<
        // UnaryOps
        daphne::EwAbsOp, daphne::EwSqrtOp, daphne::EwExpOp, daphne::EwLnOp,
        daphne::EwSinOp, daphne::EwCosOp, /*daphne::EwTanOp,
        daphne::EwAsinOp, daphne::EwAcosOp, daphne::EwAtanOp,
        daphne::EwSinhOp, daphne::EwCoshOp, daphne::EwTanhOp, */
        daphne::EwFloorOp, daphne::EwCeilOp, daphne::EwRoundOp,
        // BinaryOps
        daphne::EwAddOp, daphne::EwSubOp,
        daphne::EwMulOp, daphne::EwDivOp,
        daphne::EwPowOp>();

    populateLowerEwOpConversionPatterns(typeConverter, patterns);

    auto module = getOperation();
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
        signalPassFailure();
}

std::unique_ptr<mlir::Pass> mlir::daphne::createEwOpLoweringPass() { return std::make_unique<EwOpLoweringPass>(); }
