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

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

// TODO(phil): Look into buildLoopNest() for loop generation

constexpr int ROW = 0;
constexpr int COL = 1;

void affineMatMul(mlir::Value &lhs, mlir::Value &rhs, mlir::Value &output,
                  ConversionPatternRewriter &rewriter, mlir::Location loc,
                  ArrayRef<int64_t> lhsShape, ArrayRef<int64_t> rhsShape,
                  mlir::MLIRContext *ctx) {
    SmallVector<Value, 4> loopIvs;

    // row loop
    auto rowLoop = rewriter.create<AffineForOp>(loc, 0, lhsShape[ROW], 1);
    for (Operation &nested : *rowLoop.getBody()) {
        rewriter.eraseOp(&nested);
    }

    // row loop body
    rewriter.setInsertionPointToStart(rowLoop.getBody());

    // fma loop
    auto innerLoop = rewriter.create<AffineForOp>(loc, 0, rhsShape[ROW], 1);
    for (Operation &nested : *innerLoop.getBody()) {
        rewriter.eraseOp(&nested);
    }
    rewriter.setInsertionPointToStart(innerLoop.getBody());

    // col loop
    auto colLoop = rewriter.create<AffineForOp>(loc, 0, rhsShape[COL], 1);
    for (Operation &nested : *colLoop.getBody()) {
        rewriter.eraseOp(&nested);
    }

    // col loop body
    rewriter.setInsertionPointToStart(colLoop.getBody());

    loopIvs.push_back(rowLoop.getInductionVar());
    loopIvs.push_back(colLoop.getInductionVar());
    loopIvs.push_back(innerLoop.getInductionVar());

    // load
    mlir::Value a = rewriter.create<memref::LoadOp>(
        loc, lhs, ValueRange{loopIvs[0], loopIvs[2]});
    mlir::Value b = rewriter.create<memref::LoadOp>(
        loc, rhs, ValueRange{loopIvs[2], loopIvs[1]});
    mlir::Value c = rewriter.create<memref::LoadOp>(
        loc, output, ValueRange{loopIvs[0], loopIvs[1]});

    // fma
    mlir::Value fma = rewriter.create<LLVM::FMAOp>(loc, a, b, c);

    // store
    rewriter.create<memref::StoreOp>(loc, fma, output,
                                     ValueRange{loopIvs[0], loopIvs[1]});

    // AffineYieldOp at end of loop blocks
    rewriter.setInsertionPointToEnd(rowLoop.getBody());
    rewriter.create<AffineYieldOp>(loc);
    rewriter.setInsertionPointToEnd(colLoop.getBody());
    rewriter.create<AffineYieldOp>(loc);
    rewriter.setInsertionPointToEnd(innerLoop.getBody());
    rewriter.create<AffineYieldOp>(loc);
    rewriter.setInsertionPointAfter(rowLoop);
}

class MatMulOpLowering : public OpConversionPattern<daphne::MatMulOp> {
   public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        daphne::MatMulOp op, OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override {
        auto loc = op->getLoc();
        mlir::daphne::MatrixType lhsTensor =
            adaptor.getLhs().getType().dyn_cast<mlir::daphne::MatrixType>();
        mlir::daphne::MatrixType rhsTensor =
            adaptor.getRhs().getType().dyn_cast<mlir::daphne::MatrixType>();

        auto lhsRows = lhsTensor.getNumRows();
        auto lhsCols = lhsTensor.getNumCols();

        auto rhsRows = rhsTensor.getNumRows();
        auto rhsCols = rhsTensor.getNumCols();

        auto tensorType = lhsTensor.getElementType();
        auto lhsMemRefType =
            mlir::MemRefType::get({lhsRows, lhsCols}, tensorType);
        auto rhsMemRefType =
            mlir::MemRefType::get({rhsRows, rhsCols}, tensorType);

        mlir::MemRefType outputMemRefType =
            mlir::MemRefType::get({lhsRows, rhsCols}, tensorType);

        // daphne::Matrix -> memref
        mlir::Value lhs = rewriter.create<mlir::daphne::GetMemRefDenseMatrix>(
            op->getLoc(), lhsMemRefType, adaptor.getLhs());
        mlir::Value rhs = rewriter.create<mlir::daphne::GetMemRefDenseMatrix>(
            op->getLoc(), rhsMemRefType, adaptor.getRhs());

        // Pure memref MatMul for testing purposes
        // mlir::Value lhs = rewriter.create<memref::AllocOp>(loc, memRefType);
        // mlir::Value rhs = rewriter.create<memref::AllocOp>(loc, memRefType);
        // affineFillMemRef(5.0, rewriter, loc, nR, nC, op->getContext(), lhs);
        // affineFillMemRef(3.0, rewriter, loc, nR, nC, op->getContext(), rhs);

        // Alloc output memref
        mlir::Value outputMemRef = insertAllocAndDealloc(outputMemRefType, loc, rewriter);

        // Fill the output MemRef
        affineFillMemRef(0.0, rewriter, loc, outputMemRefType.getShape(),
                         op->getContext(), outputMemRef, tensorType);
        // Do the actual MatMul with hand built codegen
        affineMatMul(lhs, rhs, outputMemRef, rewriter, loc,
                     lhsMemRefType.getShape(), rhsMemRefType.getShape(),
                     op->getContext());

        mlir::Value DM = getDenseMatrixFromMemRef(loc, rewriter, outputMemRef, op.getType());
        rewriter.replaceOp(op, DM);
        return success();
    }
};


class SumAllOpLowering : public OpConversionPattern<daphne::AllAggSumOp> {
   public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        daphne::AllAggSumOp op, OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override {
        mlir::daphne::MatrixType tensor =
            adaptor.getArg().getType().dyn_cast<mlir::daphne::MatrixType>();

        auto loc = op->getLoc();
        auto nR = tensor.getNumRows();
        auto nC = tensor.getNumCols();

        auto tensorType = tensor.getElementType();
        auto memRefType = mlir::MemRefType::get({nR, nC}, tensorType);
        auto memRef = rewriter.create<mlir::daphne::GetMemRefDenseMatrix>(
            op->getLoc(), memRefType, adaptor.getArg());

        Value sum = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));

        SmallVector<Value, 4> loopIvs;
        // SmallVector<scf::ForOp, 2> forOps;
        SmallVector<AffineForOp, 2> forOps;
        // auto lowerBound = rewriter.create<ConstantIndexOp>(loc, 0);
        // auto outerUpperBound =
        //     rewriter.create<ConstantIndexOp>(loc, memRefShape[0]);
        // auto step = rewriter.create<ConstantIndexOp>(loc, 1);
        // outer loop
        // auto outerLoop = rewriter.create<scf::ForOp>(
        auto outerLoop =
            rewriter.create<AffineForOp>(loc, 0, nR, 1, ValueRange{sum});
        for (Operation &nested : *outerLoop.getBody()) {
            rewriter.eraseOp(&nested);
        }
        loopIvs.push_back(outerLoop.getInductionVar());
        // outer loop body
        rewriter.setInsertionPointToStart(outerLoop.getBody());
        Value sum_iter = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));
        // inner loop
        // auto innerUpperBound =
        //     rewriter.create<ConstantIndexOp>(loc, memRefShape[1]);
        // auto innerLoop = rewriter.create<scf::ForOp>(
        auto innerLoop =
            rewriter.create<AffineForOp>(loc, 0, nC, 1, ValueRange{sum_iter});
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
        mlir::Value inner_sum = rewriter.create<mlir::arith::AddFOp>(
            loc, innerLoop.getRegionIterArgs()[0], elementLoad);
        // yield inner loop result
        rewriter.setInsertionPointToEnd(innerLoop.getBody());
        // rewriter.create<scf::YieldOp>(loc, inner_sum);
        rewriter.create<AffineYieldOp>(loc, inner_sum);
        // yield outer loop result
        rewriter.setInsertionPointToEnd(outerLoop.getBody());
        mlir::Value outer_sum = rewriter.create<mlir::arith::AddFOp>(
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
        registry.insert<mlir::LLVM::LLVMDialect, mlir::AffineDialect,
                        mlir::memref::MemRefDialect>();
    }
    void runOnOperation() final;
};
}  // end anonymous namespace

Type convertFloat(mlir::FloatType floatType) {
  return IntegerType::get(floatType.getContext(),
                          floatType.getIntOrFloatBitWidth());
}


Type convertInteger(mlir::IntegerType intType) {
  return IntegerType::get(intType.getContext(),
                          intType.getIntOrFloatBitWidth());
}

llvm::Optional<Value> materializeCastFromIllegal(OpBuilder& builder, Type type,
                                                 ValueRange inputs,
                                                 Location loc) {
  Type fromType = getElementTypeOrSelf(inputs[0].getType());
  Type toType = getElementTypeOrSelf(type);
  if ((!fromType.isSignedInteger() && !fromType.isUnsignedInteger()) ||
      !toType.isSignlessInteger())
    return std::nullopt;
  // Use unrealized conversion casts to do signful->signless conversions.
  return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0])
      ->getResult(0);
}

llvm::Optional<Value> materializeCastToIllegal(OpBuilder& builder, Type type,
                                               ValueRange inputs,
                                               Location loc) {
  Type fromType = getElementTypeOrSelf(inputs[0].getType());
  Type toType = getElementTypeOrSelf(type);
  if (!fromType.isSignlessInteger() ||
      (!toType.isSignedInteger() && !toType.isUnsignedInteger()))
    return std::nullopt;
  // Use unrealized conversion casts to do signless->signful conversions.
  return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0])
      ->getResult(0);
}

void LowerDenseMatrixPass::runOnOperation() {
    mlir::ConversionTarget target(getContext());
    mlir::RewritePatternSet patterns(&getContext());
    LowerToLLVMOptions llvmOptions(&getContext());
    LLVMTypeConverter typeConverter(&getContext(), llvmOptions);

    target.addLegalDialect<mlir::memref::MemRefDialect>();
    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addLegalDialect<mlir::scf::SCFDialect>();
    target.addLegalDialect<mlir::AffineDialect>();
    target.addLegalDialect<mlir::linalg::LinalgDialect>();
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();

    target.addLegalOp<mlir::daphne::GetMemRefDenseMatrix>();
    target.addLegalOp<mlir::daphne::GetDenseMatrixFromMemRef>();
    target.addLegalOp<mlir::daphne::PrintMemRef>();
    target.addIllegalOp<mlir::daphne::AllAggSumOp>();
    target.addIllegalOp<mlir::daphne::MatMulOp>();
    typeConverter.addConversion(convertInteger);
    typeConverter.addConversion(convertFloat);
    typeConverter.addConversion([](Type type) { return type; });
    typeConverter.addArgumentMaterialization(materializeCastFromIllegal);
    typeConverter.addSourceMaterialization(materializeCastToIllegal);
    typeConverter.addTargetMaterialization(materializeCastFromIllegal);

    // typeConverter.addConversion([&](daphne::MatrixType t) {
    //     return mlir::MemRefType::get({t.getNumRows(), t.getNumCols()},
    //                                  t.getElementType());
    // });

    patterns.insert<MatMulOpLowering, SumAllOpLowering>(
        &getContext());
    auto module = getOperation();
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<mlir::Pass> mlir::daphne::createLowerDenseMatrixPass() {
    return std::make_unique<LowerDenseMatrixPass>();
}
