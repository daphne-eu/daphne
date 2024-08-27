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

#include "llvm/ADT/ArrayRef.h"
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
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
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

        mlir::Location loc = op->getLoc();

        mlir::daphne::MatrixType matrixType = adaptor.getArg().getType().dyn_cast<mlir::daphne::MatrixType>();
        mlir::Type matrixElementType = matrixType.getElementType();
        ssize_t numRows = matrixType.getNumRows();
        ssize_t numCols = matrixType.getNumCols();

        [[maybe_unused]] mlir::Value argMemref = rewriter.create<mlir::daphne::ConvertDenseMatrixToMemRef>(loc,
            mlir::MemRefType::get({numRows, numCols}, matrixElementType),
            adaptor.getArg()
        );

        Value resMemref = rewriter.create<mlir::memref::AllocOp>(loc,
            mlir::MemRefType::get({numCols, numRows}, matrixElementType)
        );

        // Affine Transpose
        auto outerLoop = rewriter.create<AffineForOp>(loc, 0, numRows, 1);
        rewriter.setInsertionPointToStart(outerLoop.getBody());
        {
            auto innerLoop = rewriter.create<AffineForOp>(loc, 0, numCols, 1);
            rewriter.setInsertionPointToStart(innerLoop.getBody());
            {
                Value argVal = rewriter.create<AffineLoadOp>(loc,
                    argMemref,
                    /*loopIvs*/ ValueRange{outerLoop.getInductionVar(), innerLoop.getInductionVar()}
                );

                rewriter.create<AffineStoreOp>(loc,
                    argVal,
                    resMemref,
                    ValueRange{innerLoop.getInductionVar(), outerLoop.getInductionVar()}
                );
            }
        }
        rewriter.setInsertionPointAfter(outerLoop);
        auto resDenseMatrix = convertMemRefToDenseMatrix(loc, rewriter, resMemref, op.getType());


        // Linalg Transpose
        // auto permutation = rewriter.getDenseI64ArrayAttr({1, 0});
        // [[maybe_unused]] ArrayRef<NamedAttribute> attributes = {};
        // auto resTranspMemref = rewriter.create<linalg::TransposeOp>(loc, argMemref, resMemref, permutation, attributes)->getResult(0);
        // auto resDenseMatrix = convertMemRefToDenseMatrix(loc, rewriter, resTranspMemref, op.getType());
        
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
struct TransposeLoweringPass : public mlir::PassWrapper<TransposeLoweringPass,
                                                    mlir::OperationPass<mlir::ModuleOp>> {
    explicit TransposeLoweringPass() {}

    StringRef getArgument() const final { return "lower-transpose"; }
    StringRef getDescription() const final {
        return "Lowers Transpose operators to a set of affine loops and performs "
               "the aggregation on a MemRef which is created from the input "
               "DenseMatrix.";
    }

    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<mlir::LLVM::LLVMDialect, mlir::AffineDialect,
                        mlir::linalg::LinalgDialect,
                        mlir::memref::MemRefDialect>();
    }
    void runOnOperation() final;
    };
}

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
    target.addLegalDialect<daphne::DaphneDialect>();
    target.addLegalDialect<mlir::AffineDialect>();
    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    target.addLegalDialect<mlir::linalg::LinalgDialect>();
    target.addLegalDialect<mlir::memref::MemRefDialect>();

    target.addLegalOp<mlir::daphne::ConvertDenseMatrixToMemRef>();
    target.addLegalOp<mlir::daphne::ConvertMemRefToDenseMatrix>();

    target.addIllegalOp<mlir::daphne::TransposeOp>();

    patterns.insert<TransposeOpLowering>(typeConverter, &getContext());
    auto module = getOperation();
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<mlir::Pass> mlir::daphne::createTransposeOpLoweringPass() {
    return std::make_unique<TransposeLoweringPass>();
}