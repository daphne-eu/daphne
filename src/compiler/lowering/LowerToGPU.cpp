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

#include "compiler/utils/LoweringUtils.h"
#include <compiler/utils/CompilerUtils.h>

#include <memory>
#include <utility>
#include <vector>

#include "api/cli/DaphneUserConfig.h"
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

struct MatMulGPU : public OpRewritePattern<daphne::MatMulOp> {
  using OpRewritePattern<daphne::MatMulOp>::OpRewritePattern;

  mutable llvm::SmallDenseSet<mlir::Value> optimized;
  LogicalResult matchAndRewrite(daphne::MatMulOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    mlir::daphne::MatrixType lhsMatrixType =
        op.getLhs().getType().dyn_cast<mlir::daphne::MatrixType>();
    mlir::daphne::MatrixType rhsMatrixType =
        op.getRhs().getType().dyn_cast<mlir::daphne::MatrixType>();

    auto lhsRows = lhsMatrixType.getNumRows();
    auto lhsCols = lhsMatrixType.getNumCols();

    auto rhsRows = rhsMatrixType.getNumRows();
    auto rhsCols = rhsMatrixType.getNumCols();

    auto matrixElementType = lhsMatrixType.getElementType();

    auto lhsMemRefType =
        mlir::MemRefType::get({lhsRows, lhsCols}, matrixElementType);
    auto rhsMemRefType =
        mlir::MemRefType::get({rhsRows, rhsCols}, matrixElementType);

    mlir::MemRefType outputMemRefType =
        mlir::MemRefType::get({lhsRows, rhsCols}, matrixElementType);

    // daphne::Matrix -> memref
    mlir::Value lhs = rewriter.create<mlir::daphne::ConvertDenseMatrixToMemRef>(
        op->getLoc(), lhsMemRefType, op.getLhs());
    mlir::Value rhs = rewriter.create<mlir::daphne::ConvertDenseMatrixToMemRef>(
        op->getLoc(), rhsMemRefType, op.getRhs());
    // mlir::Value lhs = rewriter.create<mlir::memref::AllocaOp>(loc, outputMemRefType);
    // mlir::Value rhs = rewriter.create<mlir::memref::AllocaOp>(loc, outputMemRefType);

    // Alloc output memref
    mlir::Value outputMemRef =
        insertMemRefAlloc(outputMemRefType, loc, rewriter);

    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF32Type(), rewriter.getF32FloatAttr(0));

    // Value one = rewriter.create<arith::ConstantOp>(
    //     loc, rewriter.getF32Type(), rewriter.getF32FloatAttr(1));
    // rewriter.create<linalg::FillOp>(loc, ValueRange{one}, ValueRange{lhs});
    // rewriter.create<linalg::FillOp>(loc, ValueRange{one}, ValueRange{rhs});

    rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{outputMemRef});

    mlir::Type unranked = UnrankedMemRefType::get(matrixElementType, 0);

    Value lhsC = rewriter.create<memref::CastOp>(loc, unranked, lhs);
    Value rhsC = rewriter.create<memref::CastOp>(loc, unranked, rhs);
    Value outC = rewriter.create<memref::CastOp>(loc, unranked, outputMemRef);

    rewriter.create<gpu::HostRegisterOp>(loc, lhsC);
    rewriter.create<gpu::HostRegisterOp>(loc, rhsC);
    rewriter.create<gpu::HostRegisterOp>(loc, outC);

    rewriter.create<linalg::MatmulOp>(loc,
                                      ValueRange{lhs, rhs}, ValueRange{outputMemRef});

    mlir::Value DM = convertMemRefToDenseMatrix(loc, rewriter, outputMemRef,
                                                op.getType());
    rewriter.replaceOp(op, DM);
    return mlir::success();
  }
};

struct GPULoweringPass
    : public PassWrapper<GPULoweringPass, OperationPass<ModuleOp>> {
  GPULoweringPass() = default;
  void runOnOperation() final;
  StringRef getArgument() const final { return "gpu"; }
  StringRef getDescription() const final { return ""; }
};

void GPULoweringPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.insert<MatMulGPU>(&getContext());

  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass>
daphne::createGPULoweringPass(const DaphneUserConfig &cfg) {
  return std::make_unique<GPULoweringPass>();
}
