/*
 * Copyright 2024 The DAPHNE Consortium
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
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

class TransposeOpLowering : public OpConversionPattern<daphne::TransposeOp> {
  public:
    using OpConversionPattern::OpConversionPattern;

    explicit TransposeOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
        : mlir::OpConversionPattern<daphne::TransposeOp>(typeConverter, ctx, PatternBenefit(1)) {
        this->setDebugName("TransposeOpLowering");
    }

    /**
     * @brief Replaces a Transpose operation if possible.
     * The arg Matrix is converted to a MemRef.
     * Affine loops iterate over the Memref and load/store
     * values using AffineLoad/AffineStore.
     * The result is then converted into a DenseMatrix and returned.
     *
     * @return mlir::success if Transpose has been replaced, else mlir::failure.
     */
    LogicalResult matchAndRewrite(daphne::TransposeOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {

        daphne::MatrixType matrixType = adaptor.getArg().getType().dyn_cast<daphne::MatrixType>();
        if (!matrixType) {
            return failure();
        }

        Location loc = op->getLoc();

        Type matrixElementType = matrixType.getElementType();
        ssize_t numRows = matrixType.getNumRows();
        ssize_t numCols = matrixType.getNumCols();

        if (numRows < 0 || numCols < 0) {
            return rewriter.notifyMatchFailure(
                op, "aggAll codegen currently can not handle matrix dimensions that are not known at compile time");
        }

        Value argMemref = rewriter.create<daphne::ConvertDenseMatrixToMemRef>(
            loc, MemRefType::get({numRows, numCols}, matrixElementType), adaptor.getArg());

        Value resMemref = rewriter.create<memref::AllocOp>(loc, MemRefType::get({numCols, numRows}, matrixElementType));

        auto permutation = rewriter.getDenseI64ArrayAttr({1, 0});
        rewriter.create<linalg::TransposeOp>(loc, argMemref, resMemref, permutation);

        Value resDenseMatrix = convertMemRefToDenseMatrix(loc, rewriter, resMemref, op.getType());

        rewriter.replaceOp(op, resDenseMatrix);

        return success();
    }
};

namespace {
/**
 * @brief Lowers the daphne::Transpose operator to a set of affine loops and
 * performs the aggregation using a MemRef which is created from the input
 * DenseMatrix.
 *
 * This rewrite may enable loop fusion of the produced affine loops by
 * running the loop fusion pass.
 */
struct TransposeLoweringPass : public mlir::PassWrapper<TransposeLoweringPass, mlir::OperationPass<mlir::ModuleOp>> {
    explicit TransposeLoweringPass() {}

    StringRef getArgument() const final { return "lower-transpose"; }
    StringRef getDescription() const final {
        return "Lowers Transpose operators to a set of affine loops and performs "
               "the aggregation on a MemRef which is created from the input "
               "DenseMatrix.";
    }

    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<mlir::LLVM::LLVMDialect, mlir::linalg::LinalgDialect, mlir::memref::MemRefDialect>();
    }
    void runOnOperation() final;
};
} // namespace

void TransposeLoweringPass::runOnOperation() {
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

    target.addLegalDialect<BuiltinDialect>();
    target.addLegalDialect<daphne::DaphneDialect, AffineDialect, arith::ArithDialect, LLVM::LLVMDialect,
                           linalg::LinalgDialect, memref::MemRefDialect>();

    target.addLegalOp<daphne::ConvertDenseMatrixToMemRef>();
    target.addLegalOp<daphne::ConvertMemRefToDenseMatrix>();

    target.addIllegalOp<daphne::TransposeOp>();

    patterns.insert<TransposeOpLowering>(typeConverter, &getContext());
    auto module = getOperation();
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<mlir::Pass> daphne::createTransposeOpLoweringPass() {
    return std::make_unique<TransposeLoweringPass>();
}