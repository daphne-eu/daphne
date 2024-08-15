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

class SumColOpLowering : public OpConversionPattern<daphne::ColAggSumOp> {
public:
    using OpConversionPattern::OpConversionPattern;

    explicit SumColOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
            : mlir::OpConversionPattern<daphne::ColAggSumOp>(typeConverter, ctx, PatternBenefit(1)) {
        this->setDebugName("SumColOpLowering");
    }

    /**
     * @brief Replaces a sumCol operation if possible.
     * The arg Matrix is converted to a MemRef.
     * Affine loops iterate over the Memref and load/store
     * values using AffineLoad/AffineStore.
     * The result is then converted into a DenseMatrix and returned.
     *
     * @return mlir::success if sumCol has been replaced, else mlir::failure.
     */
    LogicalResult matchAndRewrite(daphne::ColAggSumOp op, OpAdaptor adaptor,
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
            mlir::MemRefType::get({1, numCols}, matrixElementType)
        );

        // Depending on the value type, different Arith operations are needed.
        // Signed Integer values need to be converted to be unsigned before applying these ArithOps.
        if (matrixElementType.isIntOrIndex()) {
            auto outerLoop = rewriter.create<AffineForOp>(loc, 0, numCols, 1);
            rewriter.setInsertionPointToStart(outerLoop.getBody());
            {
                IntegerType signlessType = rewriter.getIntegerType(matrixElementType.getIntOrFloatBitWidth());
                Value colSum = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(signlessType));

                auto innerLoop = rewriter.create<AffineForOp>(loc, 0, numRows, 1, ValueRange{colSum});                
                rewriter.setInsertionPointToStart(innerLoop.getBody());
                {
                    // Load the next value from argMemref[j, i] and convert it to signlessType
                    Value currentElem = rewriter.create<AffineLoadOp>(loc,
                        argMemref,
                        /*loopIvs*/ ValueRange{innerLoop.getInductionVar(), outerLoop.getInductionVar()}
                    );
                    currentElem = this->typeConverter->materializeTargetConversion(rewriter, loc,
                        signlessType,
                        ValueRange{currentElem}
                    );

                    Value runningSum = rewriter.create<arith::AddIOp>(loc, innerLoop.getRegionIterArgs()[0], currentElem);
                    rewriter.create<AffineYieldOp>(loc, runningSum);
                }
                rewriter.setInsertionPointAfter(innerLoop);

                // Store the result in resMemref[0, i] after converting back to the original type
                auto castedRes = this->typeConverter->materializeTargetConversion(rewriter, loc,
                    matrixElementType,
                    ValueRange{innerLoop.getResult(0)}
                );
                rewriter.create<AffineStoreOp>(loc,
                    castedRes,
                    resMemref,
                    /*loopIvs*/ ValueRange{rewriter.create<arith::ConstantIndexOp>(loc, 0), outerLoop.getInductionVar()}
                );
            }
            rewriter.setInsertionPointAfter(outerLoop);

        } else if (matrixElementType.isF64() || matrixElementType.isF32()) {
            auto outerLoop = rewriter.create<AffineForOp>(loc, 0, numCols, 1);
            rewriter.setInsertionPointToStart(outerLoop.getBody());
            {
                Value colSum = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(matrixElementType));

                auto innerLoop = rewriter.create<AffineForOp>(loc, 0, numRows, 1, ValueRange{colSum});
                rewriter.setInsertionPointToStart(innerLoop.getBody());
                {
                    // Load the next value from argMemref[j, i]
                    Value currentElem = rewriter.create<AffineLoadOp>(loc,
                        argMemref,
                        /*loopIvs*/ ValueRange{innerLoop.getInductionVar(), outerLoop.getInductionVar()}
                    );

                    Value runningSum = rewriter.create<arith::AddFOp>(loc, innerLoop.getRegionIterArgs()[0], currentElem);
                    rewriter.create<AffineYieldOp>(loc, runningSum);
                }
                rewriter.setInsertionPointAfter(innerLoop);

                // Store the result in resMemref[0, i]
                rewriter.create<AffineStoreOp>(loc,
                    innerLoop.getResult(0),
                    resMemref,
                    /*loopIvs*/ ValueRange{rewriter.create<arith::ConstantIndexOp>(loc, 0), outerLoop.getInductionVar()}
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
 * @brief Lowers the daphne::AggCol operator to a set of affine loops and
 * performs the aggregation using a MemRef which is created from the input
 * DenseMatrix.
 *
 * This rewrite may enable loop fusion of the produced affine loops by
 * running the loop fusion pass.
 */
struct AggColLoweringPass : public mlir::PassWrapper<AggColLoweringPass,
                                                    mlir::OperationPass<mlir::ModuleOp>> {
    explicit AggColLoweringPass() {}

    StringRef getArgument() const final { return "lower-agg-col"; }
    StringRef getDescription() const final {
        return "Lowers AggCol operators to a set of affine loops and performs "
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

void AggColLoweringPass::runOnOperation() {
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

    target.addIllegalOp<mlir::daphne::ColAggSumOp>();

    patterns.insert<SumColOpLowering>(typeConverter, &getContext());
    auto module = getOperation();
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<mlir::Pass> mlir::daphne::createAggColOpLoweringPass() {
    return std::make_unique<AggColLoweringPass>();
}