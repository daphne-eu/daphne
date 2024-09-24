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

#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"
#include <compiler/utils/CompilerUtils.h>

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>
#include <utility>
#include <vector>

using namespace mlir;

class MatMulOpLowering : public OpConversionPattern<daphne::MatMulOp> {
  public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(daphne::MatMulOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        Value lhs = op.getLhs();
        Value rhs = op.getRhs();
        if (auto to = lhs.getDefiningOp<daphne::TransposeOp>()) {
            bool rhsTransposed = CompilerUtils::constantOrThrow<bool>(
                op.getTransb(), "MatMulOp.getTransb() is expected to be a constant");
            if (to.getArg() == rhs && !rhsTransposed) {
                // `t(M) @ M` -> `syrk(M)`
                rewriter.replaceOpWithNewOp<daphne::SyrkOp>(op, op.getResult().getType(), rhs);
                return success();
            }
            auto rhsMatTy = rhs.getType().dyn_cast<daphne::MatrixType>();
            if ((!rhsTransposed && rhsMatTy.getNumCols() == 1) || (rhsTransposed && rhsMatTy.getNumRows() == 1)) {
                // `t(M) @ v` -> `gemv(M, v)`
                rewriter.replaceOpWithNewOp<daphne::GemvOp>(op, op.getResult().getType(), to.getArg(), rhs);
                return success();
            }
        }
        return failure();
    }
};

namespace file_local {
struct PhyOperatorSelectionPass : public PassWrapper<PhyOperatorSelectionPass, OperationPass<ModuleOp>> {
    explicit PhyOperatorSelectionPass() {}
    void runOnOperation() final;
};
} // namespace file_local

void file_local::PhyOperatorSelectionPass::runOnOperation() {
    auto module = getOperation();

    ConversionTarget target(getContext());
    target.addLegalOp<ModuleOp>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<daphne::DaphneDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addDynamicallyLegalOp<daphne::MatMulOp>([](daphne::MatMulOp op) {
        // Note: The canonicalization of MatMulOp factors in transposed inputs:
        // - `t(X) @_ta_tb Y` to `X @_!ta_tb Y`
        // - `X @_ta_tb t(Y)` to `X @_ta_!tb Y`
        // I.e., the arguments transa and transb of MatMulOp represent if
        // the inputs shall be transposed.
        // However, we currently don't do this for the left-hand-side argument
        // (see MatMulOp::canonicalize()), once we do it there again, we need
        // to account for it here (and above in MatMulOpLowering).
        auto to = op.getLhs().getDefiningOp<daphne::TransposeOp>();
        bool rhsTransposed =
            CompilerUtils::constantOrThrow<bool>(op.getTransb(), "MatMulOp.getTransb() is expected to be a constant");
        auto rhsMatTy = op.getRhs().getType().dyn_cast<daphne::MatrixType>();
        return !(to &&
                 (
                     // `t(M) @ M` -> `syrk(M)`
                     (to.getArg() == op.getRhs() && !rhsTransposed) ||
                     // `t(M) @ v` -> `gemv(M, v)`
                     (!rhsTransposed && rhsMatTy.getNumCols() == 1) || (rhsTransposed && rhsMatTy.getNumRows() == 1)));
    });

    RewritePatternSet patterns(&getContext());
    patterns.insert<MatMulOpLowering>(&getContext());

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
        signalPassFailure();
}

std::unique_ptr<Pass> daphne::createPhyOperatorSelectionPass() {
    return std::make_unique<file_local::PhyOperatorSelectionPass>();
}