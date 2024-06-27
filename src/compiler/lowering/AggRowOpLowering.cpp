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

// #include "compiler/utils/CompilerUtils.h"
#include "compiler/utils/LoweringUtils.h"
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
// #include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
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
// #include "llvm/ADT/APFloat.h"
// #include "llvm/ADT/ArrayRef.h"

using namespace mlir;

class SumRowOpLowering : public OpConversionPattern<daphne::RowAggSumOp> {
public:
    using OpConversionPattern::OpConversionPattern;

    SumRowOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
            : mlir::OpConversionPattern<daphne::RowAggSumOp>(typeConverter, ctx) {
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

        MemRefType memRefType = mlir::MemRefType::get({numRows, numCols}, matrixElementType);
        auto memRef = rewriter.create<mlir::daphne::ConvertDenseMatrixToMemRef>(
            op->getLoc(), memRefType, adaptor.getArg());

        if (matrixElementType.isIntOrIndex()) {
            IntegerType signlessType = rewriter.getIntegerType(matrixElementType.getIntOrFloatBitWidth());
            // Reserve memory for result
            [[maybe_unused]] memref::AllocOp res = rewriter.create<memref::AllocOp>(loc, MemRefType::get({numRows, 1}, signlessType));

            // ...

            return success();
        } else if (matrixElementType.isF64() || matrixElementType.isF32()) {
            // Reserve memory for result
            memref::AllocOp res = rewriter.create<memref::AllocOp>(loc, MemRefType::get({numRows, 1}, matrixElementType));

            // Create index mappings for input and res
            // Identity map
            AffineMap inputMap = AffineMap::get(2, 0,
                                    {rewriter.getAffineDimExpr(0),
                                    rewriter.getAffineDimExpr(1)},
                                    getContext());
            // i, j -> i, 0
            AffineMap outputMap = AffineMap::get(2, 0,
                                    {rewriter.getAffineDimExpr(0),
                                    rewriter.getAffineConstantExpr(0)},
                                    getContext());
            SmallVector<AffineMap, 2> indexingMaps = {inputMap, outputMap};
            SmallVector<utils::IteratorType, 2> iteratorTypes = {utils::IteratorType::parallel, utils::IteratorType::reduction};

            // Build linalg generic op
            rewriter.create<linalg::GenericOp>(
                loc,
                res.getType(),
                memRef, // convert to ValueRange ?
                res,
                indexingMaps,
                iteratorTypes,
                [&](OpBuilder &nestedOpBuilder, Location nestedLoc, ValueRange arg) {
                    Value cumSum = nestedOpBuilder.create<arith::AddFOp>(nestedLoc, arg[0], arg[1]);
                    nestedOpBuilder.create<linalg::YieldOp>(nestedLoc, cumSum);
                } // , missing attributes ?
            );

            rewriter.create<daphne::DecRefOp>(loc, adaptor.getArg());
            rewriter.replaceOp(op, res->getResults());

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

    StringRef getArgument() const final { return "lower-agg"; }
    StringRef getDescription() const final {
        return "Lowers AggRow operators to a set of affine loops and performs "
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