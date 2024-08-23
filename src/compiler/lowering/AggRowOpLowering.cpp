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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
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

class SumRowOpLowering : public OpConversionPattern<daphne::RowAggSumOp> {
public:
    using OpConversionPattern::OpConversionPattern;

    explicit SumRowOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
            : mlir::OpConversionPattern<daphne::RowAggSumOp>(typeConverter, ctx, PatternBenefit(1)) {
        this->setDebugName("SumRowOpLowering");
    }

    /**
     * @brief Replaces a sumRow operation if possible.
     * The arg Matrix is converted to a MemRef.
     * Affine loops iterate over the Memref and load/store
     * values using AffineLoad/AffineStore.
     * The result is then converted into a DenseMatrix and returned.
     *
     * @return mlir::success if sumCol has been replaced, else mlir::failure.
     */
    LogicalResult matchAndRewrite(daphne::RowAggSumOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const override {

        mlir::Location loc = op->getLoc();

        mlir::daphne::MatrixType matrixType = adaptor.getArg().getType().dyn_cast<mlir::daphne::MatrixType>();
        mlir::Type matrixElementType = matrixType.getElementType();
        ssize_t numRows = matrixType.getNumRows();
        ssize_t numCols = matrixType.getNumCols();

        mlir::Value argMemref = rewriter.create<mlir::daphne::ConvertDenseMatrixToMemRef>(loc,
            mlir::MemRefType::get({numRows, numCols}, matrixElementType),
            adaptor.getArg()
        );

        Value resMemref = rewriter.create<mlir::memref::AllocOp>(loc,
            mlir::MemRefType::get({numRows, 1}, matrixElementType)
        );

        // Depending on the value type, different Arith operations are needed.
        // Signed Integer values need to be converted to be unsigned before applying these ArithOps.
        if (matrixElementType.isIntOrIndex()) {
            auto outerLoop = rewriter.create<AffineForOp>(loc, 0, numRows, 1);
            rewriter.setInsertionPointToStart(outerLoop.getBody());
            {
                IntegerType signlessType = rewriter.getIntegerType(matrixElementType.getIntOrFloatBitWidth());
                Value rowSum = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(signlessType));

                auto innerLoop = rewriter.create<AffineForOp>(loc, 0, numCols, 1, ValueRange{rowSum});                
                rewriter.setInsertionPointToStart(innerLoop.getBody());
                {
                    // Load the next value from argMemref[i, j] and convert it to signlessType
                    Value currentElem = rewriter.create<AffineLoadOp>(loc,
                        argMemref,
                        /*loopIvs*/ ValueRange{outerLoop.getInductionVar(), innerLoop.getInductionVar()}
                    );
                    currentElem = this->typeConverter->materializeTargetConversion(rewriter, loc,
                        signlessType,
                        ValueRange{currentElem}
                    );

                    Value runningSum = rewriter.create<arith::AddIOp>(loc, innerLoop.getRegionIterArgs()[0], currentElem);
                    rewriter.create<AffineYieldOp>(loc, runningSum);
                }
                rewriter.setInsertionPointAfter(innerLoop);

                // Store the result in resMemref[i, 0] after converting back to the original type
                auto castedRes = this->typeConverter->materializeTargetConversion(rewriter, loc,
                    matrixElementType,
                    ValueRange{innerLoop.getResult(0)}
                );
                rewriter.create<AffineStoreOp>(loc,
                    castedRes,
                    resMemref,
                    /*loopIvs*/ ValueRange{outerLoop.getInductionVar(), rewriter.create<arith::ConstantIndexOp>(loc, 0)}
                );
            }
            rewriter.setInsertionPointAfter(outerLoop);

        } else if (matrixElementType.isF64() || matrixElementType.isF32()) {
            auto outerLoop = rewriter.create<AffineForOp>(loc, 0, numRows, 1);
            rewriter.setInsertionPointToStart(outerLoop.getBody());
            {
                Value rowSum = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(matrixElementType));

                auto innerLoop = rewriter.create<AffineForOp>(loc, 0, numCols, 1, ValueRange{rowSum});
                rewriter.setInsertionPointToStart(innerLoop.getBody());
                {
                    // Load the next value from argMemref[i, j]
                    Value currentElem = rewriter.create<AffineLoadOp>(loc,
                        argMemref,
                        /*loopIvs*/ ValueRange{outerLoop.getInductionVar(), innerLoop.getInductionVar()}
                    );

                    Value runningSum = rewriter.create<arith::AddFOp>(loc, innerLoop.getRegionIterArgs()[0], currentElem);
                    rewriter.create<AffineYieldOp>(loc, runningSum);
                }
                rewriter.setInsertionPointAfter(innerLoop);

                // Store the result in resMemref[i, 0]
                rewriter.create<AffineStoreOp>(loc,
                    innerLoop.getResult(0),
                    resMemref,
                    /*loopIvs*/ ValueRange{outerLoop.getInductionVar(), rewriter.create<arith::ConstantIndexOp>(loc, 0)}
                );
            }
            rewriter.setInsertionPointAfter(outerLoop);

        } else {
            return failure();
        }

        auto resDenseMatrix = convertMemRefToDenseMatrix(loc, rewriter, resMemref, op.getType());
        rewriter.replaceOp(op, resDenseMatrix);

        return success();
    }
};

namespace {
/**
 * @brief Lowers the daphne::AggRow operator to a set of affine loops and
 * performs the aggregation using a MemRef which is created from the input
 * DenseMatrix.
 *
 * This rewrite may enable loop fusion of the produced affine loops by
 * running the loop fusion pass.
 */
struct AggRowLoweringPass : public mlir::PassWrapper<AggRowLoweringPass,
                                                    mlir::OperationPass<mlir::ModuleOp>> {
    explicit AggRowLoweringPass() {}

    StringRef getArgument() const final { return "lower-agg-row"; }
    StringRef getDescription() const final {
        return "Lowers AggRow operators to a set of affine loops and performs "
               "the aggregation on a MemRef which is created from the input "
               "DenseMatrix.";
    }

    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<mlir::LLVM::LLVMDialect, mlir::AffineDialect,
                        mlir::arith::ArithDialect, mlir::memref::MemRefDialect>();
    }
    void runOnOperation() final;
    };
}

void AggRowLoweringPass::runOnOperation() {
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
    target.addLegalDialect<mlir::memref::MemRefDialect>();

    target.addLegalOp<mlir::daphne::ConvertDenseMatrixToMemRef>();
    target.addLegalOp<mlir::daphne::ConvertMemRefToDenseMatrix>();

    target.addIllegalOp<mlir::daphne::RowAggSumOp>();

    patterns.insert<SumRowOpLowering>(typeConverter, &getContext());
    auto module = getOperation();
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<mlir::Pass> mlir::daphne::createAggRowOpLoweringPass() {
    return std::make_unique<AggRowLoweringPass>();
}