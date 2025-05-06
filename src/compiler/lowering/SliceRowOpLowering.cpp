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

//template<class SliceOp>
class SliceRowOpLowering : public OpConversionPattern<daphne::SliceRowOp> {
  public:
    using OpConversionPattern::OpConversionPattern;

    explicit SliceRowOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
        : mlir::OpConversionPattern<daphne::SliceRowOp>(typeConverter, ctx, PatternBenefit(1)) {
        this->setDebugName("SliceRowOpLowering");
    }

    /**
     * @brief Replaces a Transpose operation with a Linalg TransposeOp if possible.
     *
     * @return mlir::success if Transpose has been replaced, else mlir::failure.
     */
    LogicalResult matchAndRewrite(daphne::SliceRowOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {

        daphne::MatrixType matrixType = adaptor.getSource().getType().dyn_cast<daphne::MatrixType>();
        if (!matrixType) {
            return failure();
        }

        Location loc = op->getLoc();

        Type matrixElementType = matrixType.getElementType();
        ssize_t numRows = matrixType.getNumRows();
        ssize_t numCols = matrixType.getNumCols();

        if (numRows < 0 || numCols < 0) {
            return rewriter.notifyMatchFailure(
                op, "sliceRowOp codegen currently only works with matrix dimensions that are known at compile time");
        }

        Value argMemref = rewriter.create<daphne::ConvertDenseMatrixToMemRef>(
            loc, MemRefType::get({numRows, numCols}, matrixElementType), adaptor.getSource());

        auto lowerIncl = adaptor.getLowerIncl().getDefiningOp<daphne::ConstantOp>().getValue().dyn_cast<mlir::IntegerAttr>().getSInt();
        auto upperExcl = adaptor.getUpperExcl().getDefiningOp<daphne::ConstantOp>().getValue().dyn_cast<mlir::IntegerAttr>().getSInt();
        
        // Value resMemref = rewriter.create<memref::AllocOp>(loc, MemRefType::get({(upperExcl-lowerIncl), numCols}, matrixElementType));
        
        DenseI64ArrayAttr offset = rewriter.getDenseI64ArrayAttr({lowerIncl, 0});
        DenseI64ArrayAttr sizes = rewriter.getDenseI64ArrayAttr({(upperExcl-lowerIncl), numCols});
        DenseI64ArrayAttr strides = rewriter.getDenseI64ArrayAttr({1, 1});
        
        // Value selMemref = rewriter.create<memref::SubViewOp>(loc, argMemref, offset, sizes, strides);
        Value resMemref = rewriter.create<memref::SubViewOp>(loc, argMemref, offset, sizes, strides);

        // SmallVector<AffineMap, 2> indexMaps{AffineMap::getMultiDimIdentityMap(2, rewriter.getContext()),
        //                                     AffineMap::getMultiDimIdentityMap(2, rewriter.getContext())};

        // SmallVector<utils::IteratorType, 2> iterTypes{utils::IteratorType::parallel,
        //                                               utils::IteratorType::parallel};

        // rewriter.create<linalg::GenericOp>(loc, TypeRange{}, ValueRange{selMemref}, ValueRange{resMemref},
        //                                    indexMaps, iterTypes,
        //                                    [&](OpBuilder &OpBuilderNested, Location locNested, ValueRange arg) {
        //                                        OpBuilderNested.create<linalg::YieldOp>(locNested, arg[0]);
        //                                    });
        
        Value resDenseMatrix = convertMemRefToDenseMatrix(loc, rewriter, resMemref, op.getType());

        rewriter.replaceOp(op, resDenseMatrix);

        return success();
    }
};

namespace {
/**
 * @brief Lowers the daphne::Transpose operator to a Linalg TransposeOp.
 *
 * This rewrite may enable loop fusion on the affine loops TransposeOp is
 * lowered to by running the loop fusion pass.
 */
struct SliceRowLoweringPass : public mlir::PassWrapper<SliceRowLoweringPass, mlir::OperationPass<mlir::ModuleOp>> {
    explicit SliceRowLoweringPass() {}

    StringRef getArgument() const final { return "lower-slice-row"; }
    StringRef getDescription() const final { return "Lowers SliceRow operators to a Memref SubViewOp."; }

    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<mlir::LLVM::LLVMDialect, mlir::linalg::LinalgDialect, mlir::memref::MemRefDialect>();
    }
    void runOnOperation() final;
};
} // end anonymous namespace

void SliceRowLoweringPass::runOnOperation() {
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

    target.addLegalDialect<BuiltinDialect, daphne::DaphneDialect, linalg::LinalgDialect, memref::MemRefDialect>();

    target.addDynamicallyLegalOp<daphne::SliceRowOp>([](Operation *op) {
        Type operand = op->getOperand(0).getType();
        daphne::MatrixType matType = operand.dyn_cast<daphne::MatrixType>();
        if (matType && matType.getRepresentation() == daphne::MatrixRepresentation::Dense) {
            return false;
        }
        return true;
    });

    patterns.insert<SliceRowOpLowering>(typeConverter, &getContext());
    auto module = getOperation();
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<mlir::Pass> daphne::createSliceRowOpLoweringPass() {
    return std::make_unique<SliceRowLoweringPass>();
}