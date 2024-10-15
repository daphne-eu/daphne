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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
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

using namespace mlir;

class SumAllOpLowering : public OpConversionPattern<daphne::AllAggSumOp> {
  public:
    using OpConversionPattern::OpConversionPattern;

    SumAllOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
        : OpConversionPattern<daphne::AllAggSumOp>(typeConverter, ctx) {
        this->setDebugName("SumAllOpLowering");
    }

    LogicalResult matchAndRewrite(daphne::AllAggSumOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        daphne::MatrixType matrixType = adaptor.getArg().getType().dyn_cast<daphne::MatrixType>();
        Location loc = op->getLoc();
        ssize_t numRows = matrixType.getNumRows();
        ssize_t numCols = matrixType.getNumCols();

        Type matrixElementType = matrixType.getElementType();
        auto memRefType = MemRefType::get({numRows, numCols}, matrixElementType);
        auto argMemRef =
            rewriter.create<daphne::ConvertDenseMatrixToMemRef>(op->getLoc(), memRefType, adaptor.getArg());

        // Create a single element Memref to store the running sum in.
        // This is necessary because Linalg only accepts shaped variadics.
        Value accumulator = rewriter.create<memref::AllocaOp>(loc, MemRefType::get({1}, matrixElementType));
        rewriter.create<memref::StoreOp>(
            loc, rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(matrixElementType)), accumulator,
            ValueRange{rewriter.create<arith::ConstantIndexOp>(loc, 0)});

        SmallVector<AffineMap, 2> indexMap = {AffineMap::getMultiDimIdentityMap(2, rewriter.getContext()),
                                              AffineMap::get(2, 0, {rewriter.getAffineConstantExpr(0)}, getContext())};
        SmallVector<utils::IteratorType, 2> iterTypes = {utils::IteratorType::reduction,
                                                         utils::IteratorType::reduction};
        rewriter.create<linalg::GenericOp>(
            loc, TypeRange{}, ValueRange{argMemRef}, ValueRange{accumulator}, indexMap, iterTypes,
            [&](OpBuilder &OpBuilderNested, Location locNested, ValueRange arg) {
                Value currentElem = OpBuilderNested.create<memref::LoadOp>(
                    loc, accumulator, ValueRange{rewriter.create<arith::ConstantIndexOp>(loc, 0)});
                Value runningSum =
                    llvm::isa<IntegerType>(matrixElementType)
                        ? OpBuilderNested.create<arith::AddIOp>(locNested, currentElem, arg[0]).getResult()
                        : OpBuilderNested.create<arith::AddFOp>(locNested, currentElem, arg[0]).getResult();
                OpBuilderNested.create<linalg::YieldOp>(locNested, runningSum);
            });

        rewriter.replaceOp(op, ValueRange{rewriter.create<memref::LoadOp>(
                                   loc, accumulator, ValueRange{rewriter.create<arith::ConstantIndexOp>(loc, 0)})});

        return success();
    }
};

namespace {
/**
 * @brief Lowers the daphne::AggAll operator to a Linalg GenericOp
 * which iterates over a MemRef that is created from the input DenseMatrix
 * and uses a single element Memref to store the aggregation result.
 *
 * This rewrite may enable loop fusion of the GenericOp or lowered Affine
 * loops using the loop fusion pass.
 */
struct AggAllLoweringPass : public PassWrapper<AggAllLoweringPass, OperationPass<ModuleOp>> {
    explicit AggAllLoweringPass() {}

    StringRef getArgument() const final { return "lower-agg"; }
    StringRef getDescription() const final {
        return "Lowers AggAll operators to a set of affine loops and performs "
               "the aggregation on a MemRef which is created from the input "
               "DenseMatrix.";
    }

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<LLVM::LLVMDialect, AffineDialect, memref::MemRefDialect, linalg::LinalgDialect>();
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
    typeConverter.addArgumentMaterialization(materializeCastFromIllegal);
    typeConverter.addSourceMaterialization(materializeCastToIllegal);
    typeConverter.addTargetMaterialization(materializeCastFromIllegal);

    target.addLegalDialect<AffineDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<BuiltinDialect>();
    target.addLegalDialect<daphne::DaphneDialect>();
    target.addLegalDialect<linalg::LinalgDialect>();
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<scf::SCFDialect>();

    target.addIllegalOp<daphne::AllAggSumOp>();

    patterns.insert<SumAllOpLowering>(typeConverter, &getContext());
    auto module = getOperation();
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> daphne::createAggAllOpLoweringPass() { return std::make_unique<AggAllLoweringPass>(); }
