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

#include "compiler/utils/LoweringUtils.h"
#include <util/ErrorHandler.h>

#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
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
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

// ****************************************************************************
// AggAllOp templates
// ****************************************************************************

/**
 * @brief template for lowering fully aggregating functions.
 * Aggregation is initialized with the first value of the input MemRef.
 * A Linalg GenericOp then iterates over the remainder of the first row
 * and the remaining (n-1) x m matrix. The next element as well as the
 * running aggregation result are mapped using the corresponding SI/UI/FOp
 * to update the result.
 *
 * @param AggOp The target operation this pass aims to rewrite.
 * @param SIOp The binary operation applied along the axis for signed integers.
 * @param UIOp The binary operation applied along the axis for unsigned integers.
 * @param FOp The binary operation applied along the axis for floating point values.
 */
template <typename AggOp, typename SIOp, typename UIOp, typename FOp>
class AggAllOpLowering : public OpConversionPattern<AggOp> {
  public:
    using OpAdaptor = typename OpConversionPattern<AggOp>::OpAdaptor;

    AggAllOpLowering(TypeConverter &typeConverter, MLIRContext *ctx) : OpConversionPattern<AggOp>(typeConverter, ctx) {
        this->setDebugName("AggAllOpLowering");
    }

    LogicalResult matchAndRewrite(AggOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {

        daphne::MatrixType matrixType = llvm::dyn_cast<daphne::MatrixType>(adaptor.getArg().getType());
        if (!matrixType) {
            return failure();
        }

        Location loc = op->getLoc();
        ssize_t numRows = matrixType.getNumRows();
        ssize_t numCols = matrixType.getNumCols();

        if (numRows < 0 || numCols < 0) {
            throw ErrorHandler::compilerError(
                loc, "AggAllOpLowering",
                "aggAllOp codegen currently only works with matrix dimensions that are known at compile time");
        }

        Type matrixElementType = matrixType.getElementType();
        MemRefType memRefType = MemRefType::get({numRows, numCols}, matrixElementType);
        auto argMemRef = rewriter.create<daphne::ConvertDenseMatrixToMemRef>(loc, memRefType, adaptor.getArg());

        // Create a singleton Memref to store the running aggregation result in.
        // This is necessary because Linalg only accepts shaped variadics.
        // Store first elem of argMemRef into accumulator and then iterate over remainder.
        Value accumulator = rewriter.create<memref::AllocaOp>(loc, MemRefType::get({1}, matrixElementType));
        Value initValue = rewriter.create<memref::LoadOp>(loc, argMemRef,
                                                          ValueRange{rewriter.create<arith::ConstantIndexOp>(loc, 0),
                                                                     rewriter.create<arith::ConstantIndexOp>(loc, 0)});
        rewriter.create<memref::StoreOp>(loc, initValue, accumulator,
                                         ValueRange{rewriter.create<arith::ConstantIndexOp>(loc, 0)});

        SmallVector<AffineMap, 2> indexMap{
            AffineMap::getMultiDimIdentityMap(2, rewriter.getContext()),
            AffineMap::get(2, 0, {rewriter.getAffineConstantExpr(0)}, rewriter.getContext())};
        SmallVector<utils::IteratorType, 2> iterTypes{utils::IteratorType::reduction, utils::IteratorType::reduction};

        // Aggregate over the remainder of the first row of argMemRef before aggregating over the remaining values.
        SmallVector<OpFoldResult, 2> firstRowOffsets{rewriter.getIndexAttr(0), rewriter.getIndexAttr(1)};
        SmallVector<OpFoldResult, 2> firstRowSizes{rewriter.getIndexAttr(1), rewriter.getIndexAttr(numCols - 1)};
        SmallVector<OpFoldResult, 2> firstRowStrides{rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)};

        Value firstRow =
            rewriter.create<memref::SubViewOp>(loc, argMemRef, firstRowOffsets, firstRowSizes, firstRowStrides);

        rewriter.create<linalg::GenericOp>(
            loc, TypeRange{}, ValueRange{firstRow}, ValueRange{accumulator}, indexMap, iterTypes,
            [&](OpBuilder &OpBuilderNested, Location locNested, ValueRange arg) {
                Value currentElem = OpBuilderNested.create<memref::LoadOp>(
                    locNested, accumulator, ValueRange{OpBuilderNested.create<arith::ConstantIndexOp>(locNested, 0)});
                Value nextElem = arg[0];
                Value runningAgg;

                if (llvm::isa<IntegerType>(matrixElementType)) {
                    currentElem = convertToSignlessInt(OpBuilderNested, locNested, this->typeConverter, currentElem,
                                                       matrixElementType);
                    nextElem = convertToSignlessInt(OpBuilderNested, locNested, this->typeConverter, nextElem,
                                                    matrixElementType);
                }

                if (matrixElementType.isSignedInteger()) {
                    runningAgg = OpBuilderNested.create<SIOp>(locNested, currentElem, nextElem).getResult();
                } else if (matrixElementType.isUnsignedInteger()) {
                    runningAgg = OpBuilderNested.create<UIOp>(locNested, currentElem, nextElem).getResult();
                } else {
                    runningAgg = OpBuilderNested.create<FOp>(locNested, currentElem, nextElem).getResult();
                }

                if (llvm::isa<IntegerType>(matrixElementType)) {
                    runningAgg = this->typeConverter->materializeTargetConversion(OpBuilderNested, locNested,
                                                                                  matrixElementType, runningAgg);
                }

                OpBuilderNested.create<linalg::YieldOp>(locNested, runningAgg);
            });

        SmallVector<OpFoldResult, 2> remainderOffsets{rewriter.getIndexAttr(1), rewriter.getIndexAttr(0)};
        SmallVector<OpFoldResult, 2> remainderSizes{rewriter.getIndexAttr(numRows - 1), rewriter.getIndexAttr(numCols)};
        SmallVector<OpFoldResult, 2> remainderStrides{rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)};

        Value remainder =
            rewriter.create<memref::SubViewOp>(loc, argMemRef, remainderOffsets, remainderSizes, remainderStrides);

        rewriter.create<linalg::GenericOp>(
            loc, TypeRange{}, ValueRange{remainder}, ValueRange{accumulator}, indexMap, iterTypes,
            [&](OpBuilder &OpBuilderNested, Location locNested, ValueRange arg) {
                Value currentElem = OpBuilderNested.create<memref::LoadOp>(
                    locNested, accumulator, ValueRange{OpBuilderNested.create<arith::ConstantIndexOp>(locNested, 0)});
                Value nextElem = arg[0];
                Value runningAgg;

                if (llvm::isa<IntegerType>(matrixElementType)) {
                    currentElem = convertToSignlessInt(OpBuilderNested, locNested, this->typeConverter, currentElem,
                                                       matrixElementType);
                    nextElem = convertToSignlessInt(OpBuilderNested, locNested, this->typeConverter, nextElem,
                                                    matrixElementType);
                }

                if (matrixElementType.isSignedInteger()) {
                    runningAgg = OpBuilderNested.create<SIOp>(locNested, currentElem, nextElem).getResult();
                } else if (matrixElementType.isUnsignedInteger()) {
                    runningAgg = OpBuilderNested.create<UIOp>(locNested, currentElem, nextElem).getResult();
                } else {
                    runningAgg = OpBuilderNested.create<FOp>(locNested, currentElem, nextElem).getResult();
                }

                if (llvm::isa<IntegerType>(matrixElementType)) {
                    runningAgg = this->typeConverter->materializeTargetConversion(OpBuilderNested, locNested,
                                                                                  matrixElementType, runningAgg);
                }

                OpBuilderNested.create<linalg::YieldOp>(locNested, runningAgg);
            });

        rewriter.replaceOp(op, ValueRange{rewriter.create<memref::LoadOp>(
                                   loc, accumulator, ValueRange{rewriter.create<arith::ConstantIndexOp>(loc, 0)})});

        return success();
    }
};

// ****************************************************************************
// AggAllOp specializations
// ****************************************************************************

using SumAllOpLowering = AggAllOpLowering<daphne::AllAggSumOp, arith::AddIOp, arith::AddIOp, arith::AddFOp>;
using MinAllOpLowering = AggAllOpLowering<daphne::AllAggMinOp, arith::MinSIOp, arith::MinUIOp, arith::MinimumFOp>;
using MaxAllOpLowering = AggAllOpLowering<daphne::AllAggMaxOp, arith::MaxSIOp, arith::MaxUIOp, arith::MaximumFOp>;

namespace {
/**
 * @brief Lowers the daphne::AllAgg operator to a Linalg GenericOp
 * which iterates over a MemRef that is created from the input DenseMatrix
 * and uses a singleton MemRef to store the aggregation result.
 *
 * This rewrite may enable loop fusion of the GenericOp or lowered Affine
 * loops using the loop fusion pass.
 */
struct AggAllLoweringPass : public PassWrapper<AggAllLoweringPass, OperationPass<ModuleOp>> {
    explicit AggAllLoweringPass() = default;

    [[nodiscard]] StringRef getArgument() const final { return "lower-agg"; }
    [[nodiscard]] StringRef getDescription() const final {
        return "Lowers AllAgg* operators to a Linalg GenericOp and performs "
               "the aggregation on a MemRef which is created from the input "
               "DenseMatrix.";
    }

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<LLVM::LLVMDialect, arith::ArithDialect, memref::MemRefDialect, linalg::LinalgDialect>();
    }
    void runOnOperation() final;
};
} // end anonymous namespace

void AggAllLoweringPass::runOnOperation() {
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    LowerToLLVMOptions llvmOptions(&getContext());
    LLVMTypeConverter typeConverter(&getContext(), llvmOptions);

    typeConverter.addConversion(convertInteger);
    typeConverter.addConversion(convertFloat);
    typeConverter.addConversion([](Type type) { return type; });
    // typeConverter.addArgumentMaterialization(materializeCastFromIllegal);
    typeConverter.addSourceMaterialization(materializeCastToIllegal);
    typeConverter.addTargetMaterialization(materializeCastFromIllegal);

    target.addLegalDialect<affine::AffineDialect, arith::ArithDialect, BuiltinDialect, daphne::DaphneDialect,
                           linalg::LinalgDialect, LLVM::LLVMDialect, memref::MemRefDialect>();

    target.addDynamicallyLegalOp<daphne::AllAggSumOp, daphne::AllAggMinOp, daphne::AllAggMaxOp>([](Operation *op) {
        Type operand = op->getOperand(0).getType();
        auto matType = llvm::dyn_cast<daphne::MatrixType>(operand);
        if (matType && matType.getRepresentation() == daphne::MatrixRepresentation::Dense) {
            return false;
        }
        return true;
    });

    patterns.insert<SumAllOpLowering, MinAllOpLowering, MaxAllOpLowering>(typeConverter, &getContext());
    auto module = getOperation();
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> daphne::createAggAllOpLoweringPass() { return std::make_unique<AggAllLoweringPass>(); }
