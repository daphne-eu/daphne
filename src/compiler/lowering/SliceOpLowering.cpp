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

template<class SliceOp, size_t sliceAlongDim>
class SliceOpLowering : public OpConversionPattern<SliceOp> {
  public:
    using OpAdaptor = typename OpConversionPattern<SliceOp>::OpAdaptor;

    explicit SliceOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
        : OpConversionPattern<SliceOp>(typeConverter, ctx, PatternBenefit(1)) {
        this->setDebugName("SliceOpLowering");
    }

    /**
     * @brief Replaces a Slice operation with a MemRef SubviewOp if possible.
     *
     * @return mlir::success if Slice has been replaced, else mlir::failure.
     */
    LogicalResult matchAndRewrite(SliceOp op, OpAdaptor adaptor,
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
                op, "sliceOp codegen currently only works with matrix dimensions that are known at compile time");
        }

        Value argMemref = rewriter.create<daphne::ConvertDenseMatrixToMemRef>(
            loc, MemRefType::get({numRows, numCols}, matrixElementType), adaptor.getSource());

        auto lowerIncl = adaptor.getLowerIncl().template getDefiningOp<daphne::ConstantOp>().getValue().template dyn_cast<mlir::IntegerAttr>().getSInt();
        auto upperExcl = adaptor.getUpperExcl().template getDefiningOp<daphne::ConstantOp>().getValue().template dyn_cast<mlir::IntegerAttr>().getSInt();
                
        DenseI64ArrayAttr offset = sliceAlongDim == ROW ? rewriter.getDenseI64ArrayAttr({lowerIncl, 0})
                                                        : rewriter.getDenseI64ArrayAttr({0, lowerIncl});

        DenseI64ArrayAttr sizes = sliceAlongDim == ROW ? rewriter.getDenseI64ArrayAttr({(upperExcl-lowerIncl), numCols})
                                                       : rewriter.getDenseI64ArrayAttr({numRows, (upperExcl-lowerIncl)});                                                
        
        DenseI64ArrayAttr strides = rewriter.getDenseI64ArrayAttr({1, 1});
        
        Value resMemref = rewriter.create<memref::SubViewOp>(loc, argMemref, offset, sizes, strides);
        
        Value resDenseMatrix = convertMemRefToDenseMatrix(loc, rewriter, resMemref, op.getType());

        rewriter.replaceOp(op, resDenseMatrix);

        return success();
    }
};

using SliceRowOpLowering = SliceOpLowering<daphne::SliceRowOp, ROW>;
using SliceColOpLowering = SliceOpLowering<daphne::SliceColOp, COL>;

namespace {
/**
 * @brief Lowers the daphne::Slice operator to a Memref SubviewOp.
 */
struct SliceLoweringPass : public mlir::PassWrapper<SliceLoweringPass, mlir::OperationPass<mlir::ModuleOp>> {
    explicit SliceLoweringPass() {}

    StringRef getArgument() const final { return "lower-slice"; }
    StringRef getDescription() const final { return "Lowers SliceRow/SliceCol operators to a Memref SubViewOp."; }

    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<mlir::LLVM::LLVMDialect, mlir::memref::MemRefDialect>();
    }
    void runOnOperation() final;
};
} // end anonymous namespace

void SliceLoweringPass::runOnOperation() {
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

    target.addLegalDialect<BuiltinDialect, daphne::DaphneDialect, memref::MemRefDialect>();

    target.addDynamicallyLegalOp<daphne::SliceRowOp, daphne::SliceColOp>([](Operation *op) {
        Type operand = op->getOperand(0).getType();
        daphne::MatrixType matType = operand.dyn_cast<daphne::MatrixType>();
        if (matType && matType.getRepresentation() == daphne::MatrixRepresentation::Dense) {
            return false;
        }
        return true;
    });

    patterns.insert<SliceRowOpLowering, SliceColOpLowering>(typeConverter, &getContext());
    auto module = getOperation();
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<mlir::Pass> daphne::createSliceOpLoweringPass() {
    return std::make_unique<SliceLoweringPass>();
}