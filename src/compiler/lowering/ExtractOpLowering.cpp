/*
 * Copyright 2025 The DAPHNE Consortium
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
#include "mlir/IR/BuiltinAttributes.h"
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
using namespace std;

static constexpr size_t ROW = 0;
static constexpr size_t COL = 1;

template<class ExtractOp, size_t extractAlongDim>
class ExtractOpLowering : public OpConversionPattern<ExtractOp> {
  public:
    //using OpConversionPattern<SliceOp>::OpConversionPattern;
    using OpAdaptor = typename OpConversionPattern<ExtractOp>::OpAdaptor;

    explicit ExtractOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
        : OpConversionPattern<ExtractOp>(typeConverter, ctx, PatternBenefit(1)) {
        this->setDebugName("ExtractOpLowering");
    }

    /**
     * @brief Replaces a Transpose operation with a Linalg TransposeOp if possible.
     *
     * @return mlir::success if Transpose has been replaced, else mlir::failure.
     */
    LogicalResult matchAndRewrite(ExtractOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {

        daphne::MatrixType matrixType = adaptor.getSource().getType().template dyn_cast<daphne::MatrixType>();
        if (!matrixType) {
            return failure();
        }

        Location loc = op->getLoc();

        Type matrixElementType = matrixType.getElementType();
        ssize_t numRows = matrixType.getNumRows();
        ssize_t numCols = matrixType.getNumCols();

        if (numRows < 0 || numCols < 0) {
            return rewriter.notifyMatchFailure(
                op, "extractOp codegen currently only works with matrix dimensions that are known at compile time");
        }

        Value argMemref = rewriter.create<daphne::ConvertDenseMatrixToMemRef>(
            loc, MemRefType::get({numRows, numCols}, matrixElementType), adaptor.getSource());

        
        daphne::MatrixType selectionType = adaptor.getSelectedRows().getType().template dyn_cast<daphne::MatrixType>();
        if (!matrixType) {
            return failure();
        }

      

        Type selectionElementType = selectionType.getElementType();
        ssize_t numSelectedRows = selectionType.getNumRows();

        Value selectionMemref = rewriter.create<daphne::ConvertDenseMatrixToMemRef>(
            loc, MemRefType::get({numSelectedRows, 1}, matrixElementType), adaptor.getSelectedRows());
        
        Value resMemref = rewriter.create<memref::AllocOp>(loc, MemRefType::get({numSelectedRows, numCols}, matrixElementType));

        for (ssize_t i = 0; i < numSelectedRows; i++)
        {

            Value des = rewriter.create<memref::SubViewOp>(loc, resMemref, 
                rewriter.getDenseI64ArrayAttr({i, 0}), 
                rewriter.getDenseI64ArrayAttr({1, numCols}), 
                rewriter.getDenseI64ArrayAttr({1, 1}));

            Value select = rewriter.create<memref::LoadOp>(loc, selectionMemref, 
                ValueRange{rewriter.create<arith::ConstantIndexOp>(loc, i),
               rewriter.create<arith::ConstantIndexOp>(loc, 0)});

            Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(0));

            ValueRange offsets = {select, zero};
            ValueRange sizes = {rewriter.create<arith::ConstantIndexOp>(loc, 1),
               rewriter.create<arith::ConstantIndexOp>(loc, numCols)};
            ValueRange strides = {rewriter.create<arith::ConstantIndexOp>(loc, 1),
               rewriter.create<arith::ConstantIndexOp>(loc, 1)};

            Value src = rewriter.create<memref::SubViewOp>(loc, argMemref, offsets, sizes, strides);

            rewriter.create<memref::CopyOp>(loc, src, des);

        }
        
        
        Value resDenseMatrix = convertMemRefToDenseMatrix(loc, rewriter, resMemref, op.getType());

        rewriter.replaceOp(op, resDenseMatrix);

        return success();
    }
};

using ExtractRowOpLowering = ExtractOpLowering<daphne::ExtractRowOp, ROW>;
//using ExtractColOpLowering = ExtractOpLowering<daphne::ExtractColOp, COL>;

namespace {
/**
 * @brief Lowers the daphne::Transpose operator to a Linalg TransposeOp.
 *
 * This rewrite may enable loop fusion on the affine loops TransposeOp is
 * lowered to by running the loop fusion pass.
 */
struct ExtractLoweringPass : public mlir::PassWrapper<ExtractLoweringPass, mlir::OperationPass<mlir::ModuleOp>> {
    explicit ExtractLoweringPass() {}

    StringRef getArgument() const final { return "lower-extract"; }
    StringRef getDescription() const final { return "Lowers ExtractRow/ExtractCol operators to a Memref SubViewOp."; }

    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<mlir::LLVM::LLVMDialect, mlir::memref::MemRefDialect, mlir::arith::ArithDialect>();
    }
    void runOnOperation() final;
};
} // end anonymous namespace

void ExtractLoweringPass::runOnOperation() {
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

    target.addLegalDialect<BuiltinDialect, daphne::DaphneDialect, memref::MemRefDialect, arith::ArithDialect>();

    target.addDynamicallyLegalOp<daphne::ExtractRowOp/*, daphne::ExtractColOp*/>([](Operation *op) {
        Type operand = op->getOperand(0).getType();
        daphne::MatrixType matType = operand.dyn_cast<daphne::MatrixType>();
        if (matType && matType.getRepresentation() == daphne::MatrixRepresentation::Dense) {
            return false;
        }
        return true;
    });

    patterns.insert<ExtractRowOpLowering/*, ExtractColOpLowering*/>(typeConverter, &getContext());
    auto module = getOperation();
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<mlir::Pass> daphne::createExtractOpLoweringPass() {
    return std::make_unique<ExtractLoweringPass>();
}