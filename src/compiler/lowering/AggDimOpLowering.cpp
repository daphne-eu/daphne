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
#include "mlir/Dialect/Affine/TransformOps/AffineTransformOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
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
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

using namespace mlir;

static constexpr size_t ROW = 0;
static constexpr size_t COL = 1;

// ****************************************************************************
// AggDimOp templates
// ****************************************************************************

/**
 * @brief template for lowering partial aggregate functions along a dimension.
 * Aggregation is initialized with the values along the 1st row or column
 * of the input MemRef. Afterwards, the next element along a row/column
 * is loaded, the corresponding binary UI/SI/F Op is applied to this element
 * and the running aggregation result in the corresponding row/column and
 * their result is stored again.
 *
 * @param AggOp The target operation this pass aims to rewrite.
 * @param SIOp The binary operation applied along the axis for signed integers.
 * @param UIOp The binary operation applied along the axis for unsigned integers.
 * @param FOp The binary operation applied along the axis for floating point values.
 * @param aggAlongDim `0` (ROW) or `1` (COL) to specify the axis along
 * which to aggregate. If the pass aggregates along a row, all columns are
 * collapsed and vice versa.
 */
template <typename AggOp, typename SIOp, typename UIOp, typename FOp, size_t aggAlongDim>
class AggDimOpLowering : public OpConversionPattern<AggOp> {
  public:
    using OpAdaptor = typename OpConversionPattern<AggOp>::OpAdaptor;

    AggDimOpLowering(TypeConverter &typeConverter, MLIRContext *ctx) : OpConversionPattern<AggOp>(typeConverter, ctx) {
        this->setDebugName("AggDimOpLowering");
    }

    LogicalResult matchAndRewrite(AggOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {

        daphne::MatrixType matrixType = adaptor.getArg().getType().template dyn_cast<daphne::MatrixType>();
        if (!matrixType) {
            return failure();
        }

        Location loc = op->getLoc();
        ssize_t numRows = matrixType.getNumRows();
        ssize_t numCols = matrixType.getNumCols();

        if (numRows < 0 || numCols < 0) {
            throw ErrorHandler::compilerError(
                loc, "AggDimOpLowering",
                "aggDimOp codegen currently only works with matrix dimensions that are known at compile time");
        }

        Type matrixElementType = matrixType.getElementType();
        MemRefType argMemRefType = MemRefType::get({numRows, numCols}, matrixElementType);
        auto argMemRef = rewriter.create<daphne::ConvertDenseMatrixToMemRef>(loc, argMemRefType, adaptor.getArg());

        MemRefType resMemRefType = aggAlongDim == ROW ? MemRefType::get({numRows, 1}, matrixElementType)
                                                      : MemRefType::get({1, numCols}, matrixElementType);
        Value resMemRef = rewriter.create<memref::AllocOp>(loc, resMemRefType);

        // Fill resMemRef with first row/column of argMemRef before aggregating over the remaining values.
        SmallVector<OpFoldResult, 2> initValsOffsets{rewriter.getIndexAttr(0), rewriter.getIndexAttr(0)};
        SmallVector<OpFoldResult, 2> initValSizes =
            aggAlongDim == ROW ? SmallVector<OpFoldResult, 2>{rewriter.getIndexAttr(numRows), rewriter.getIndexAttr(1)}
                               : SmallVector<OpFoldResult, 2>{rewriter.getIndexAttr(1), rewriter.getIndexAttr(numCols)};
        SmallVector<OpFoldResult, 2> initValStrides{rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)};

        Value initValues =
            rewriter.create<memref::SubViewOp>(loc, argMemRef, initValsOffsets, initValSizes, initValStrides);

        SmallVector<AffineMap, 2> initValIndexMaps{AffineMap::getMultiDimIdentityMap(2, rewriter.getContext()),
                                                   AffineMap::getMultiDimIdentityMap(2, rewriter.getContext())};

        SmallVector<utils::IteratorType, 2> initValIterTypes{utils::IteratorType::parallel,
                                                             utils::IteratorType::parallel};

        rewriter.create<linalg::GenericOp>(loc, TypeRange{}, ValueRange{initValues}, ValueRange{resMemRef},
                                           initValIndexMaps, initValIterTypes,
                                           [&](OpBuilder &OpBuilderNested, Location locNested, ValueRange arg) {
                                               OpBuilderNested.create<linalg::YieldOp>(locNested, arg[0]);
                                           });

        // Aggregate over the remaining values.
        SmallVector<OpFoldResult, 2> remainderOffsets =
            aggAlongDim == ROW ? SmallVector<OpFoldResult, 2>{rewriter.getIndexAttr(0), rewriter.getIndexAttr(1)}
                               : SmallVector<OpFoldResult, 2>{rewriter.getIndexAttr(1), rewriter.getIndexAttr(0)};
        SmallVector<OpFoldResult, 2> remainderSizes =
            aggAlongDim == ROW
                ? SmallVector<OpFoldResult, 2>{rewriter.getIndexAttr(numRows), rewriter.getIndexAttr(numCols - 1)}
                : SmallVector<OpFoldResult, 2>{rewriter.getIndexAttr(numRows - 1), rewriter.getIndexAttr(numCols)};
        SmallVector<OpFoldResult, 2> remainderStrides{rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)};

        Value remainderValues =
            rewriter.create<memref::SubViewOp>(loc, argMemRef, remainderOffsets, remainderSizes, remainderStrides);

        SmallVector<AffineMap, 2> remainderIndexMaps = {AffineMap::getMultiDimIdentityMap(2, rewriter.getContext())};
        AffineMap indexMapRes =
            aggAlongDim == ROW ? AffineMap::get(2, 0, {rewriter.getAffineDimExpr(0), rewriter.getAffineConstantExpr(0)},
                                                rewriter.getContext())
                               : AffineMap::get(2, 0, {rewriter.getAffineConstantExpr(0), rewriter.getAffineDimExpr(1)},
                                                rewriter.getContext());
        remainderIndexMaps.push_back(indexMapRes);

        SmallVector<utils::IteratorType, 2> remainderIterTypes =
            aggAlongDim == ROW
                ? SmallVector<utils::IteratorType, 2>{utils::IteratorType::parallel, utils::IteratorType::reduction}
                : SmallVector<utils::IteratorType, 2>{utils::IteratorType::reduction, utils::IteratorType::parallel};

        rewriter.create<linalg::GenericOp>(
            loc, TypeRange{}, ValueRange{remainderValues}, ValueRange{resMemRef}, remainderIndexMaps,
            remainderIterTypes, [&](OpBuilder &OpBuilderNested, Location locNested, ValueRange arg) {
                ValueRange resIndex = aggAlongDim == ROW
                                          ? ValueRange{OpBuilderNested.create<linalg::IndexOp>(locNested, 0),
                                                       OpBuilderNested.create<arith::ConstantIndexOp>(locNested, 0)}
                                          : ValueRange{OpBuilderNested.create<arith::ConstantIndexOp>(locNested, 0),
                                                       OpBuilderNested.create<linalg::IndexOp>(locNested, 1)};
                Value storedElem = OpBuilderNested.create<memref::LoadOp>(locNested, resMemRef, resIndex);
                Value currentElem = arg[0];

                if (llvm::isa<IntegerType>(matrixElementType)) {
                    currentElem = convertToSignlessInt(OpBuilderNested, locNested, this->typeConverter, currentElem,
                                                       matrixElementType);
                    storedElem = convertToSignlessInt(OpBuilderNested, locNested, this->typeConverter, storedElem,
                                                      matrixElementType);

                    storedElem = matrixElementType.isSignedInteger()
                                     ? OpBuilderNested.create<SIOp>(locNested, storedElem, currentElem).getResult()
                                     : OpBuilderNested.create<UIOp>(locNested, storedElem, currentElem).getResult();

                    storedElem = this->typeConverter->materializeTargetConversion(OpBuilderNested, locNested,
                                                                                  matrixElementType, storedElem);
                } else {
                    storedElem = OpBuilderNested.create<FOp>(locNested, storedElem, currentElem).getResult();
                }
                OpBuilderNested.create<linalg::YieldOp>(locNested, storedElem);
            });

        auto resDenseMatrix = convertMemRefToDenseMatrix(loc, rewriter, resMemRef, op.getType());

        rewriter.replaceOp(op, resDenseMatrix);

        return success();
    }
};

/**
 * @brief template for lowering partial aggregate functions along a dimension
 * that return an index.
 * The result is initialized with zeros. During iteration, the next element
 * along the row/column as well as the element the current result index points
 * to are loaded and the passed binary `AggOp` is applied to them. If MaxIdx
 * is `true`, the index of the greater (or equal) value is stored in the
 * corresponding row/column. If a row/column contain multiple values equal
 * to the max/min value, returns the index of the first one.
 *
 * @param AggOp The target operation this pass aims to rewrite.
 * @param MaxIdx Bool to determine whether to take the maximum (true) or
 * minimum (false) along the specified dimension.
 * @param aggAlongDim `0` (ROW) or `1` (COL) to specify the axis along
 * which to aggregate. If the pass aggregates along a row, all columns are
 * collapsed and vice versa.
 */
template <typename AggOp, bool MaxIdx, size_t aggAlongDim>
class AggDimIdxOpLowering : public OpConversionPattern<AggOp> {
  public:
    using OpAdaptor = typename OpConversionPattern<AggOp>::OpAdaptor;

    AggDimIdxOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
        : OpConversionPattern<AggOp>(typeConverter, ctx) {
        this->setDebugName("AggDimIdxOpLowering");
    }

    LogicalResult matchAndRewrite(AggOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {

        daphne::MatrixType matrixType = adaptor.getArg().getType().template dyn_cast<daphne::MatrixType>();
        if (!matrixType) {
            return failure();
        }

        Location loc = op->getLoc();
        ssize_t numRows = matrixType.getNumRows();
        ssize_t numCols = matrixType.getNumCols();

        if (numRows < 0 || numCols < 0) {
            throw ErrorHandler::compilerError(
                loc, "AggDimOpLowering",
                "aggDimOp codegen currently only works with matrix dimensions that are known at compile time");
        }

        Type matrixElementType = matrixType.getElementType();
        MemRefType argMemRefType = MemRefType::get({numRows, numCols}, matrixElementType);
        auto argMemRef = rewriter.create<daphne::ConvertDenseMatrixToMemRef>(loc, argMemRefType, adaptor.getArg());

        MemRefType resMemRefType = aggAlongDim == ROW ? MemRefType::get({numRows, 1}, rewriter.getIndexType())
                                                      : MemRefType::get({1, numCols}, rewriter.getIndexType());
        Value resMemRef = rewriter.create<memref::AllocOp>(loc, resMemRefType);

        // For simplicity, the inner loop always iterates over the dimension that is aggregated.
        // ResMemRef is implicitely initialized with zeros, thus iteration for the inner loop starts at the 2nd
        // row/column. Todo: optimize column aggregation for cache layout (row-major layout for dense matrices).
        ssize_t outerLoopUB = aggAlongDim == ROW ? numRows : numCols;
        ssize_t innerLoopUB = aggAlongDim == ROW ? numCols : numRows;

        auto outerLoop = rewriter.create<AffineForOp>(loc, 0, outerLoopUB, 1);
        rewriter.setInsertionPointToStart(outerLoop.getBody());
        {

            Value resIdx = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(rewriter.getIndexType()));

            // Index to load the corresponding value in argMemRef.
            ValueRange initArgValIdx = aggAlongDim == ROW ? ValueRange{outerLoop.getInductionVar(), resIdx}
                                                          : ValueRange{resIdx, outerLoop.getInductionVar()};
            Value currentResVal = rewriter.create<AffineLoadOp>(loc, argMemRef, initArgValIdx);

            auto innerLoop = rewriter.create<AffineForOp>(loc, 1, innerLoopUB, 1, ValueRange{resIdx, currentResVal});
            rewriter.setInsertionPointToStart(innerLoop.getBody());
            {
                ValueRange cmpValIdx = aggAlongDim == ROW
                                           ? ValueRange{outerLoop.getInductionVar(), innerLoop.getInductionVar()}
                                           : ValueRange{innerLoop.getInductionVar(), outerLoop.getInductionVar()};
                Value cmpVal = rewriter.create<AffineLoadOp>(loc, argMemRef, cmpValIdx);

                Value cmpResBool;
                Value currentResVal = innerLoop.getRegionIterArgs()[1];
                if (llvm::isa<IntegerType>(matrixElementType)) {
                    arith::CmpIPredicate cmpFunc;
                    if (matrixElementType.isSignedInteger()) {
                        cmpFunc = MaxIdx ? arith::CmpIPredicate::sge : arith::CmpIPredicate::sle;
                    } else {
                        cmpFunc = MaxIdx ? arith::CmpIPredicate::uge : arith::CmpIPredicate::ule;
                    }
                    currentResVal =
                        convertToSignlessInt(rewriter, loc, this->typeConverter, currentResVal, matrixElementType);
                    cmpVal = convertToSignlessInt(rewriter, loc, this->typeConverter, cmpVal, matrixElementType);
                    cmpResBool = rewriter.create<arith::CmpIOp>(loc, cmpFunc, currentResVal, cmpVal);
                } else {
                    arith::CmpFPredicate cmpFunc = MaxIdx ? arith::CmpFPredicate::OGE : arith::CmpFPredicate::OLE;
                    cmpResBool = rewriter.create<arith::CmpFOp>(loc, cmpFunc, currentResVal, cmpVal);
                }

                resIdx = rewriter.create<arith::SelectOp>(loc, cmpResBool, innerLoop.getRegionIterArgs()[0],
                                                          innerLoop.getInductionVar());
                currentResVal = rewriter.create<arith::SelectOp>(loc, cmpResBool, currentResVal, cmpVal);

                rewriter.create<AffineYieldOp>(
                    loc, ValueRange{resIdx, this->typeConverter->materializeTargetConversion(
                                                rewriter, loc, matrixElementType, currentResVal)});
            } // end inner loop
            rewriter.setInsertionPointAfter(innerLoop);

            ValueRange storeResIdx =
                aggAlongDim == ROW
                    ? ValueRange{outerLoop.getInductionVar(),
                                 rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(rewriter.getIndexType()))}
                    : ValueRange{rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(rewriter.getIndexType())),
                                 outerLoop.getInductionVar()};
            rewriter.create<AffineStoreOp>(loc, innerLoop.getResult(0), resMemRef, storeResIdx);

        } // end outer loop
        rewriter.setInsertionPointAfter(outerLoop);

        auto resDenseMatrix = convertMemRefToDenseMatrix(loc, rewriter, resMemRef, op.getType());

        rewriter.replaceOp(op, resDenseMatrix);

        return success();
    }
};

// ****************************************************************************
// AggDimOp specializations
// ****************************************************************************

using SumRowOpLowering = AggDimOpLowering<daphne::RowAggSumOp, arith::AddIOp, arith::AddIOp, arith::AddFOp, ROW>;
using SumColOpLowering = AggDimOpLowering<daphne::ColAggSumOp, arith::AddIOp, arith::AddIOp, arith::AddFOp, COL>;
using MinRowOpLowering = AggDimOpLowering<daphne::RowAggMinOp, arith::MinSIOp, arith::MinUIOp, arith::MinFOp, ROW>;
using MinColOpLowering = AggDimOpLowering<daphne::ColAggMinOp, arith::MinSIOp, arith::MinUIOp, arith::MinFOp, COL>;
using MaxRowOpLowering = AggDimOpLowering<daphne::RowAggMaxOp, arith::MaxSIOp, arith::MaxUIOp, arith::MaxFOp, ROW>;
using MaxColOpLowering = AggDimOpLowering<daphne::ColAggMaxOp, arith::MaxSIOp, arith::MaxUIOp, arith::MaxFOp, COL>;

using ArgMinRowOpLowering = AggDimIdxOpLowering<daphne::RowAggIdxMinOp, false, ROW>;
using ArgMinColOpLowering = AggDimIdxOpLowering<daphne::ColAggIdxMinOp, false, COL>;
using ArgMaxRowOpLowering = AggDimIdxOpLowering<daphne::RowAggIdxMaxOp, true, ROW>;
using ArgMaxColOpLowering = AggDimIdxOpLowering<daphne::ColAggIdxMaxOp, true, COL>;

namespace {
/**
 * @brief Lowers the daphne::RowAgg* and daphne::ColAgg* operator to a Linalg GenericOp
 * or Affine ForOp Nest which iterates over a MemRef that is created from the input DenseMatrix
 * and uses a (2 dim) single row/column Memref to store the aggregation results.
 *
 * This rewrite may enable loop fusion of the GenericOp or lowered Affine
 * loops using the loop fusion pass.
 */
struct AggDimLoweringPass : public PassWrapper<AggDimLoweringPass, OperationPass<ModuleOp>> {
    explicit AggDimLoweringPass() = default;

    [[nodiscard]] StringRef getArgument() const final { return "lower-agg-dim"; }
    [[nodiscard]] StringRef getDescription() const final {
        return "Lowers *Agg operators to a Linalg Generic Op or a "
               "set of affine loops and performs the aggregation on "
               "a MemRef which is created from the input DenseMatrix.";
    }

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<LLVM::LLVMDialect, AffineDialect, arith::ArithDialect, memref::MemRefDialect,
                        linalg::LinalgDialect, scf::SCFDialect>();
    }
    void runOnOperation() final;
};
} // end anonymous namespace

void AggDimLoweringPass::runOnOperation() {
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    LowerToLLVMOptions llvmOptions(&getContext());
    LLVMTypeConverter typeConverter(&getContext(), llvmOptions);

    typeConverter.addConversion(convertInteger);
    typeConverter.addConversion(convertFloat);
    typeConverter.addConversion([](Type type) { return type; });
    typeConverter.addArgumentMaterialization(materializeCastFromIllegal);
    typeConverter.addSourceMaterialization(materializeCastToIllegal);
    typeConverter.addTargetMaterialization(materializeCastFromIllegal);

    target.addLegalDialect<AffineDialect, arith::ArithDialect, BuiltinDialect, daphne::DaphneDialect,
                           linalg::LinalgDialect, LLVM::LLVMDialect, memref::MemRefDialect>();

    target.addDynamicallyLegalOp<daphne::RowAggSumOp, daphne::ColAggSumOp, daphne::RowAggMinOp, daphne::ColAggMinOp,
                                 daphne::RowAggMaxOp, daphne::ColAggMaxOp, daphne::RowAggIdxMinOp,
                                 daphne::ColAggIdxMinOp, daphne::RowAggIdxMaxOp, daphne::ColAggIdxMaxOp>(
        [](Operation *op) {
            Type operand = op->getOperand(0).getType();
            auto matType = operand.dyn_cast<daphne::MatrixType>();
            if (matType && matType.getRepresentation() == daphne::MatrixRepresentation::Dense) {
                return false;
            }
            return true;
        });

    patterns
        .insert<SumRowOpLowering, SumColOpLowering, MinRowOpLowering, MinColOpLowering, MaxRowOpLowering,
                MaxColOpLowering, ArgMinRowOpLowering, ArgMinColOpLowering, ArgMaxRowOpLowering, ArgMaxColOpLowering>(
            typeConverter, &getContext());
    auto module = getOperation();
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> daphne::createAggDimOpLoweringPass() { return std::make_unique<AggDimLoweringPass>(); }
