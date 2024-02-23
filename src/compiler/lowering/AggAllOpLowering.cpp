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
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"

using namespace mlir;

class SumAllOpLowering : public OpConversionPattern<daphne::AllAggSumOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  SumAllOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : mlir::OpConversionPattern<daphne::AllAggSumOp>(typeConverter, ctx) {
    this->setDebugName("SumAllOpLowering");
  }
  // Float and Integer value type matrices have to be handled separately, since
  // arith operations are different.
  LogicalResult
  matchAndRewrite(daphne::AllAggSumOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    mlir::daphne::MatrixType matrixType =
        adaptor.getArg().getType().dyn_cast<mlir::daphne::MatrixType>();
    auto loc = op->getLoc();
    auto nR = matrixType.getNumRows();
    auto nC = matrixType.getNumCols();

    auto matrixElementType = matrixType.getElementType();
    auto memRefType = mlir::MemRefType::get({nR, nC}, matrixElementType);
    auto memRef = rewriter.create<mlir::daphne::ConvertDenseMatrixToMemRef>(
        op->getLoc(), memRefType, adaptor.getArg());

    if (matrixElementType.isIntOrIndex()) {
      IntegerType signless_type =
          rewriter.getIntegerType(matrixElementType.getIntOrFloatBitWidth());
      Value sum = rewriter.create<mlir::arith::ConstantOp>(
          loc, signless_type, rewriter.getIntegerAttr(signless_type, 0));

      SmallVector<Value, 4> loopIvs;
      SmallVector<AffineForOp, 2> forOps;
      auto outerLoop =
          rewriter.create<AffineForOp>(loc, 0, nR, 1, ValueRange{sum});
      for (Operation &nested : *outerLoop.getBody()) {
        rewriter.eraseOp(&nested);
      }
      loopIvs.push_back(outerLoop.getInductionVar());
      // outer loop body
      rewriter.setInsertionPointToStart(outerLoop.getBody());
      Value sum_iter = rewriter.create<mlir::arith::ConstantOp>(
          loc, signless_type, rewriter.getIntegerAttr(signless_type, 0));
      // inner loop
      auto innerLoop =
          rewriter.create<AffineForOp>(loc, 0, nC, 1, ValueRange{sum_iter});
      for (Operation &nested : *innerLoop.getBody()) {
        rewriter.eraseOp(&nested);
      }
      loopIvs.push_back(innerLoop.getInductionVar());
      // inner loop body
      rewriter.setInsertionPointToStart(innerLoop.getBody());
      // load value from memref
      Value elementLoad = rewriter.create<memref::LoadOp>(loc, memRef, loopIvs);
      auto castedElement = this->typeConverter->materializeSourceConversion(
          rewriter, loc, signless_type, ValueRange{elementLoad});
      // sum loop iter arg and memref value
      mlir::Value inner_sum = rewriter.create<mlir::arith::AddIOp>(
          loc, innerLoop.getRegionIterArgs()[0], castedElement);
      // yield inner loop result
      rewriter.setInsertionPointToEnd(innerLoop.getBody());
      rewriter.create<AffineYieldOp>(loc, inner_sum);
      // yield outer loop result
      rewriter.setInsertionPointToEnd(outerLoop.getBody());
      mlir::Value outer_sum = rewriter.create<mlir::arith::AddIOp>(
          loc, outerLoop.getRegionIterArgs()[0], innerLoop.getResult(0));
      rewriter.create<AffineYieldOp>(loc, outer_sum);

      rewriter.setInsertionPointAfter(outerLoop);
      rewriter.create<daphne::DecRefOp>(loc, adaptor.getArg());
      // replace sumAll op with result of loops
      auto castedRes = this->typeConverter->materializeTargetConversion(
          rewriter, loc, matrixElementType,
          ValueRange{outerLoop->getResult(0)});
      rewriter.replaceOp(op, ValueRange{castedRes});

      return success();
    } else {
      Value sum = rewriter.create<mlir::arith::ConstantOp>(
          loc, matrixElementType, rewriter.getFloatAttr(matrixElementType, 0));

      SmallVector<Value, 4> loopIvs;
      SmallVector<AffineForOp, 2> forOps;
      auto outerLoop =
          rewriter.create<AffineForOp>(loc, 0, nR, 1, ValueRange{sum});
      for (Operation &nested : *outerLoop.getBody()) {
        rewriter.eraseOp(&nested);
      }
      loopIvs.push_back(outerLoop.getInductionVar());
      // outer loop body
      rewriter.setInsertionPointToStart(outerLoop.getBody());
      Value sum_iter = rewriter.create<mlir::arith::ConstantOp>(
          loc, matrixElementType, rewriter.getFloatAttr(matrixElementType, 0));
      // inner loop
      auto innerLoop =
          rewriter.create<AffineForOp>(loc, 0, nC, 1, ValueRange{sum_iter});
      for (Operation &nested : *innerLoop.getBody()) {
        rewriter.eraseOp(&nested);
      }
      loopIvs.push_back(innerLoop.getInductionVar());
      // inner loop body
      rewriter.setInsertionPointToStart(innerLoop.getBody());
      // load value from memref
      auto elementLoad = rewriter.create<memref::LoadOp>(loc, memRef, loopIvs);
      // sum loop iter arg and memref value
      mlir::Value inner_sum = rewriter.create<mlir::arith::AddFOp>(
          loc, innerLoop.getRegionIterArgs()[0], elementLoad);
      // yield inner loop result
      rewriter.setInsertionPointToEnd(innerLoop.getBody());
      rewriter.create<AffineYieldOp>(loc, inner_sum);
      // yield outer loop result
      rewriter.setInsertionPointToEnd(outerLoop.getBody());
      mlir::Value outer_sum = rewriter.create<mlir::arith::AddFOp>(
          loc, outerLoop.getRegionIterArgs()[0], innerLoop.getResult(0));
      rewriter.create<AffineYieldOp>(loc, outer_sum);

      rewriter.setInsertionPointAfter(outerLoop);
      rewriter.create<daphne::DecRefOp>(loc, adaptor.getArg());
      // replace sumAll op with result of loops
      rewriter.replaceOp(op, outerLoop.getResult(0));

      return success();
    }
  }
};

namespace {
/**
 * @brief Lowers the daphne::AggAll operator to a set of affine loops and
 * performs the aggregation on a MemRef which is created from the input
 * DenseMatrix.
 *
 * This rewrite may enable loop fusion of the produced affine loops by
 * running the loop fusion pass.
 */
struct AggAllLoweringPass
    : public mlir::PassWrapper<AggAllLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  explicit AggAllLoweringPass() {}

  StringRef getArgument() const final { return "lower-agg"; }
  StringRef getDescription() const final {
    return "Lowers AggAll operators to a set of affine loops and performs "
           "the aggregation on a MemRef which is created from the input "
           "DenseMatrix.";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect, mlir::AffineDialect,
                    mlir::memref::MemRefDialect>();
  }
  void runOnOperation() final;
};
} // end anonymous namespace

void AggAllLoweringPass::runOnOperation() {
  mlir::ConversionTarget target(getContext());
  mlir::RewritePatternSet patterns(&getContext());
  LowerToLLVMOptions llvmOptions(&getContext());
  LLVMTypeConverter typeConverter(&getContext(), llvmOptions);

  typeConverter.addConversion(convertInteger);
  typeConverter.addConversion(convertFloat);
  typeConverter.addConversion([](Type type) { return type; });
  typeConverter.addArgumentMaterialization(materializeCastFromIllegal);
  typeConverter.addSourceMaterialization(materializeCastToIllegal);
  typeConverter.addTargetMaterialization(materializeCastFromIllegal);

  target.addLegalDialect<mlir::memref::MemRefDialect>();
  target.addLegalDialect<mlir::arith::ArithDialect>();
  target.addLegalDialect<mlir::scf::SCFDialect>();
  target.addLegalDialect<mlir::AffineDialect>();
  target.addLegalDialect<mlir::linalg::LinalgDialect>();
  target.addLegalDialect<mlir::LLVM::LLVMDialect>();
  target.addLegalDialect<daphne::DaphneDialect>();
  target.addLegalDialect<BuiltinDialect>();

  target.addLegalOp<mlir::daphne::ConvertDenseMatrixToMemRef>();
  target.addLegalOp<mlir::daphne::ConvertMemRefToDenseMatrix>();
  target.addLegalOp<mlir::daphne::DecRefOp>();

  target.addIllegalOp<mlir::daphne::AllAggSumOp>();

  patterns.insert<SumAllOpLowering>(typeConverter, &getContext());
  auto module = getOperation();
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> mlir::daphne::createAggAllOpLoweringPass() {
  return std::make_unique<AggAllLoweringPass>();
}
