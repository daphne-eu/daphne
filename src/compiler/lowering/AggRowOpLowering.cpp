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

/*
#include "compiler/utils/LoweringUtils.h"
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
*/
#include <iostream>


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
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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

class SumRowOpLowering : public OpConversionPattern<daphne::RowAggSumOp> {
public:
    using OpConversionPattern::OpConversionPattern;

    explicit SumRowOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
            : mlir::OpConversionPattern<daphne::RowAggSumOp>(typeConverter, ctx,
                                                                PatternBenefit(1)) { // determine benfit
        this->setDebugName("SumRowOpLowering");
    }

    // Float and Integer value type matrices have to be handled separately, since
    // arith operations are different.
    LogicalResult matchAndRewrite(daphne::RowAggSumOp op, OpAdaptor adaptor,
            ConversionPatternRewriter &rewriter) const override {

        mlir::Location loc = op->getLoc();
        
        mlir::daphne::MatrixType matrixType = adaptor.getArg().getType().dyn_cast<mlir::daphne::MatrixType>();
        mlir::Type matrixElementType = matrixType.getElementType();
        ssize_t numRows = matrixType.getNumRows();
        ssize_t numCols = matrixType.getNumCols();

        MemRefType argMemRefType = mlir::MemRefType::get({numRows, numCols}, matrixElementType);
        [[maybe_unused]] mlir::Value argMatrix = rewriter.create<mlir::daphne::ConvertDenseMatrixToMemRef>(
            loc, argMemRefType, adaptor.getArg());

        if (matrixElementType.isIntOrIndex()) {
            [[maybe_unused]] IntegerType signlessType = rewriter.getIntegerType(matrixElementType.getIntOrFloatBitWidth());
            // Reserve memory for result

            // ...

            return success();
        } else if (matrixElementType.isF64() || matrixElementType.isF32()) {
            // Reserve memory for result
            // mlir::Value outputMemref = insertMemRefAlloc(mlir::MemRefType::get({numRows, 1}, matrixElementType), loc, rewriter);
            // mlir::Value outputMemref = rewriter.create<memref::AllocaOp>(loc, MemRefType::get({numRows, 1}, matrixElementType));
            // affineFillMemRef(0.0, rewriter, loc, {numRows, 1}, getContext(), outputMemref, matrixElementType);
            
            // attempt with linalg
            auto initOutputTensor = rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{numRows, 1}, matrixElementType).getResult();
            auto fillVal = rewriter.create<arith::ConstantOp>(loc, FloatAttr::get(matrixElementType, 0.0));
            mlir::Value outputTensor = rewriter.create<linalg::FillOp>(loc, ValueRange{fillVal}, initOutputTensor).getResult(0);

            // Create index mappings for input and res
            AffineMap inputMap = AffineMap::getMultiDimIdentityMap(2, getContext());
            // i, j -> i, 0
            AffineMap outputMap = AffineMap::get(2, 0,
                                    {rewriter.getAffineDimExpr(0),
                                    rewriter.getAffineConstantExpr(0)},
                                    getContext());
            SmallVector<AffineMap, 2> indexingMaps{inputMap, outputMap};
            SmallVector<utils::IteratorType, 2> iteratorTypes = {utils::IteratorType::parallel, utils::IteratorType::reduction};

            // Build linalg generic op
            auto genericOp = rewriter.create<linalg::GenericOp>(
                    loc,
                    // MemRefType::get({numRows, 1}, matrixElementType),
                    outputTensor.getType(),
                    ValueRange{argMatrix},
                    ValueRange{outputTensor},
                    indexingMaps,
                    iteratorTypes,
                    [&](OpBuilder &OpBuilderNested, Location locNested, ValueRange arg) {
                        Value cumSum = OpBuilderNested.create<arith::AddFOp>(locNested, arg[1], arg[0]);
                        // Value cumSum = OpBuilderNested.create<linalg::AddOp>(locNested, arg[0], arg[1]);
                        OpBuilderNested.create<linalg::YieldOp>(locNested, cumSum);
                    } // , missing attributes ?
                );

            // auto outputMemref = rewriter.create<bufferization::AllocTensorOp>(loc, MemRefType::get({numRows, 1}, matrixElementType), genericOp.getResult(0)); // RankedTensorType, ArrayRef<int64_t>
            auto outputMemref = rewriter.create<bufferization::ToMemrefOp>(loc, MemRefType::get({numRows, 1}, matrixElementType), genericOp.getResult(0));

            rewriter.create<daphne::DecRefOp>(loc, adaptor.getArg());

            auto outputDenseMatrix = convertMemRefToDenseMatrix(loc, rewriter, outputMemref, op.getType());
            // auto outputDenseMatrix = convertMemRefToDenseMatrix(loc, rewriter, outputMemref, op.getType()); // inc ref?
            rewriter.replaceOp(op, outputDenseMatrix);

            return success();
        } else {
            return failure();
        }
    }
};

namespace {
/**
 * @brief Lowers the daphne::AggRow operator to a set of affine loops and
 * performs the aggregation on a MemRef which is created from the input
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
                        mlir::memref::MemRefDialect, mlir::linalg::LinalgDialect,
                        mlir::bufferization::BufferizationDialect,
                        mlir::tensor::TensorDialect>();
    }
    void runOnOperation() final;
    };
} // end anonymous namespace

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

    target.addLegalDialect<mlir::memref::MemRefDialect>();
    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addLegalDialect<mlir::scf::SCFDialect>();
    target.addLegalDialect<mlir::AffineDialect>();
    target.addLegalDialect<mlir::linalg::LinalgDialect>();
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    target.addLegalDialect<daphne::DaphneDialect>();
    target.addLegalDialect<BuiltinDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalDialect<mlir::bufferization::BufferizationDialect>();

    target.addLegalOp<mlir::daphne::ConvertDenseMatrixToMemRef>();
    target.addLegalOp<mlir::daphne::ConvertMemRefToDenseMatrix>();
    target.addLegalOp<mlir::daphne::DecRefOp>();

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