/*
 * Copyright 2022 The DAPHNE Consortium
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

#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "compiler/utils/CompilerUtils.h"
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APFloat.h"

using namespace mlir;


// TODO(phil): Look into buildLoopNest() for loop generation

void affineFillMemRef(double value, ConversionPatternRewriter &rewriter, mlir::Location loc,
                ssize_t nR, ssize_t nC, mlir::MLIRContext *ctx, mlir::Value memRef) {
    Value cst0 = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(value));
    SmallVector<Value, 4> loopIvs;
    // TODO(phil): function for loops (fill/matmul)
    // // SmallVector<scf::ForOp, 2> forOps;
    // SmallVector<AffineForOp, 2> forOps;
    //
    auto outerLoop = rewriter.create<AffineForOp>(loc, 0, nR, 1);
    for (Operation &nested : *outerLoop.getBody()) {
        rewriter.eraseOp(&nested);
    }
    loopIvs.push_back(outerLoop.getInductionVar());

    // outer loop body
    rewriter.setInsertionPointToStart(outerLoop.getBody());
    auto innerLoop = rewriter.create<AffineForOp>(loc, 0, nC, 1);
    for (Operation &nested : *innerLoop.getBody()) {
        rewriter.eraseOp(&nested);
    }
    loopIvs.push_back(innerLoop.getInductionVar());
    rewriter.create<AffineYieldOp>(loc);
    rewriter.setInsertionPointToStart(innerLoop.getBody());
    rewriter.create<AffineStoreOp>(loc, cst0, memRef,
                                   loopIvs);
    // rewriter.create<memref::StoreOp>(loc, cst0, outputMemRef,
    // ValueRange{loopIvs});
    rewriter.create<AffineYieldOp>(loc);
    rewriter.setInsertionPointAfter(outerLoop);
}

// affine.for %arg0 = 0 to 2048 {
//     affine.for %arg1 = 0 to 2048 {
//         affine.for %arg2 = 0 to 2048 {
//             %a = affine.load %A[%arg0, %arg2] : memref<2018x2018xf61>
//             %b = affine.load %B[%arg2, %arg1] : memref<2018x2018xf61>
//             %ci = affine.load %C[%arg0, %arg1] : memref<2018x2018xf61>
//             %p = arith.mulf %a, %b : f61
//             %co = arith.addf %ci, %p : f61
//             affine.store %co, %C[%arg0, %arg1] : memref<2018x2018xf61>
//         }
//     }
// }
void affineMatMul(mlir::Value &lhs, mlir::Value &rhs, mlir::Value &output, ConversionPatternRewriter &rewriter, mlir::Location loc,
        ssize_t nR, ssize_t nC, mlir::MLIRContext *ctx) {
    SmallVector<Value, 4> loopIvs;
    Value cst0 = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));

    // row loop
    auto rowLoop = rewriter.create<AffineForOp>(loc, 0, nR, 1);
    for (Operation &nested : *rowLoop.getBody()) {
        rewriter.eraseOp(&nested);
    }
    loopIvs.push_back(rowLoop.getInductionVar());

    // row loop body
    rewriter.setInsertionPointToStart(rowLoop.getBody());

    // col loop
    auto colLoop = rewriter.create<AffineForOp>(loc, 0, nC, 1);
    for (Operation &nested : *colLoop.getBody()) {
        rewriter.eraseOp(&nested);
    }
    loopIvs.push_back(colLoop.getInductionVar());

    // col loop body
    rewriter.setInsertionPointToStart(colLoop.getBody());

    // fma loop
    auto innerLoop = rewriter.create<AffineForOp>(loc, 0, nR, 1);
    for (Operation &nested : *innerLoop.getBody()) {
        rewriter.eraseOp(&nested);
    }
    loopIvs.push_back(innerLoop.getInductionVar());
    rewriter.setInsertionPointToStart(innerLoop.getBody());

    // load
    mlir::Value a = rewriter.create<memref::LoadOp>(loc, lhs, ValueRange{loopIvs[0], loopIvs[2]}); // ivs[0, 2]
    mlir::Value b = rewriter.create<memref::LoadOp>(loc, rhs, ValueRange{loopIvs[2], loopIvs[1]});
    mlir::Value c = rewriter.create<memref::LoadOp>(loc, output, ValueRange{loopIvs[0], loopIvs[1]});

    // fma
    mlir::Value fma = rewriter.create<LLVM::FMAOp>(loc, a, b, c);

    // store
    rewriter.create<memref::StoreOp>(loc, fma, output, ValueRange{loopIvs[0], loopIvs[1]});

    // AffineYieldOp at end of loop blocks
    rewriter.setInsertionPointToEnd(rowLoop.getBody());
    rewriter.create<AffineYieldOp>(loc);
    rewriter.setInsertionPointToEnd(colLoop.getBody());
    rewriter.create<AffineYieldOp>(loc);
    rewriter.setInsertionPointToEnd(innerLoop.getBody());
    rewriter.create<AffineYieldOp>(loc);
    rewriter.setInsertionPointAfter(rowLoop);
}

class MatMulOpLowering : public OpConversionPattern<daphne::MatMulOp> {
   public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        daphne::MatMulOp op, OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override {
        auto loc = op->getLoc();
        mlir::daphne::MatrixType tensor =
            adaptor.getLhs().getType().dyn_cast<mlir::daphne::MatrixType>();
        auto nR = tensor.getNumRows();
        auto nC = tensor.getNumCols();
        auto tensorType = tensor.getElementType();
        auto memRefType = mlir::MemRefType::get(
            {nR, nC}, tensorType, {mlir::AffineMap::getMinorIdentityMap(2, 2, op->getContext())});
        MemRefLayoutAttrInterface t;
        std::cout << "tensorType: " << std::endl;
        tensorType.dump();
        std::cout << "memRefType: " << std::endl;
        memRefType.dump();

        // daphne::Matrix -> memref
        mlir::Value lhs = rewriter.create<mlir::daphne::GetMemRefDenseMatrix>(
                op->getLoc(), memRefType, adaptor.getLhs());
        mlir::Value rhs = rewriter.create<mlir::daphne::GetMemRefDenseMatrix>(
            op->getLoc(), memRefType, adaptor.getRhs());
        // Pure memref MatMul
        // mlir::Value lhs = rewriter.create<memref::AllocOp>(loc, memRefType);
        // mlir::Value rhs = rewriter.create<memref::AllocOp>(loc, memRefType);
        // affineFillMemRef(3.0, rewriter, loc, nR, nC, op->getContext(), lhs);
        // affineFillMemRef(3.0, rewriter, loc, nR, nC, op->getContext(), rhs);


        // Alloc output memref
        mlir::Value outputMemRef = rewriter.create<memref::AllocOp>(loc, memRefType);

        // Fill the output MemRef
        affineFillMemRef(0.0, rewriter, loc, nR, nC, op->getContext(), outputMemRef);
        // Do the actual MatMul with hand built codegen
        affineMatMul(lhs, rhs, outputMemRef, rewriter, loc, nR, nC, op->getContext());

        auto extractStridedMetadataOp =
            rewriter.create<memref::ExtractStridedMetadataOp>(loc, outputMemRef);
        // Base ptr.
        mlir::Value basePtr = extractStridedMetadataOp.getBaseBuffer();
        mlir::Value alignedPtr = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, outputMemRef);
        // offset.
        mlir::Value offset = extractStridedMetadataOp.getOffset();
        // strides.
        mlir::ResultRange strides = extractStridedMetadataOp.getStrides();
        // sizes
        mlir::ResultRange sizes = extractStridedMetadataOp.getSizes();
        // IREE approach to sizes
        // mlir::Value size0 = rewriter.create<memref::DimOp>(loc, outputMemRef, 0);
        // mlir::Value size1 = rewriter.create<memref::DimOp>(loc, outputMemRef, 1);

        rewriter.create<mlir::daphne::PrintMemRef>(loc, alignedPtr, offset, sizes[0], sizes[1], strides[0], strides[1]);
        // rewriter.create<mlir::daphne::PrintMemRef>(loc, basePtr, offset, size0, size1, strides[0], strides[1]);
        // rewriter.create<mlir::daphne::PrintMemRef>(loc, basePtr);

        // TODO(phil): MemRefDescriptor in MemRefBuilder.h may be useful

        mlir::Value DM = rewriter.create<mlir::daphne::GetDenseMatrixFromMemRef>(
                // loc, op.getType(), outputMemRef);
                loc, op.getType(), alignedPtr, offset, sizes[0], sizes[1], strides[0], strides[1]);
        rewriter.replaceOp(op, DM);
        return success();
    }
};

class SumAllOpLowering : public OpConversionPattern<daphne::AllAggSumOp> {
   public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        daphne::AllAggSumOp op, OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override {
        mlir::daphne::MatrixType tensor =
            adaptor.getArg().getType().dyn_cast<mlir::daphne::MatrixType>();

        auto loc = op->getLoc();
        auto nR = tensor.getNumRows();
        auto nC = tensor.getNumCols();

        auto tensorType = tensor.getElementType();
        auto memRefType = mlir::MemRefType::get(
            {nR, nC}, tensorType);
        auto memRef = rewriter.create<mlir::daphne::GetMemRefDenseMatrix>(
            op->getLoc(), memRefType, adaptor.getArg());

        Value sum = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));

        SmallVector<Value, 4> loopIvs;
        // SmallVector<scf::ForOp, 2> forOps;
        SmallVector<AffineForOp, 2> forOps;
        // auto lowerBound = rewriter.create<ConstantIndexOp>(loc, 0);
        // auto outerUpperBound =
        //     rewriter.create<ConstantIndexOp>(loc, memRefShape[0]);
        // auto step = rewriter.create<ConstantIndexOp>(loc, 1);
        // outer loop
        // auto outerLoop = rewriter.create<scf::ForOp>(
        auto outerLoop = rewriter.create<AffineForOp>(
            loc, 0, nR, 1, ValueRange{sum});
        for (Operation &nested : *outerLoop.getBody()) {
            rewriter.eraseOp(&nested);
        }
        loopIvs.push_back(outerLoop.getInductionVar());
        // outer loop body
        rewriter.setInsertionPointToStart(outerLoop.getBody());
        Value sum_iter = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));
        // inner loop
        // auto innerUpperBound =
        //     rewriter.create<ConstantIndexOp>(loc, memRefShape[1]);
        // auto innerLoop = rewriter.create<scf::ForOp>(
        auto innerLoop = rewriter.create<AffineForOp>(
            loc, 0, nC, 1, ValueRange{sum_iter});
        for (Operation &nested : *innerLoop.getBody()) {
            rewriter.eraseOp(&nested);
        }
        loopIvs.push_back(innerLoop.getInductionVar());
        // inner loop body
        rewriter.setInsertionPointToStart(innerLoop.getBody());
        // load value from memref
        auto elementLoad =
            rewriter.create<memref::LoadOp>(loc, memRef, loopIvs);
        // sum loop iter arg and memref value
        mlir::Value inner_sum = rewriter.create<mlir::arith::AddFOp>(
            loc, innerLoop.getRegionIterArgs()[0], elementLoad);
        // yield inner loop result
        rewriter.setInsertionPointToEnd(innerLoop.getBody());
        // rewriter.create<scf::YieldOp>(loc, inner_sum);
        rewriter.create<AffineYieldOp>(loc, inner_sum);
        // yield outer loop result
        rewriter.setInsertionPointToEnd(outerLoop.getBody());
        mlir::Value outer_sum = rewriter.create<mlir::arith::AddFOp>(
            loc, outerLoop.getRegionIterArgs()[0], innerLoop.getResult(0));
        // rewriter.create<scf::YieldOp>(loc, outer_sum);
        rewriter.create<AffineYieldOp>(loc, outer_sum);

        // replace sumAll op with result of loops
        rewriter.replaceOp(op, outerLoop.getResult(0));
        return success();
    }
};

namespace {
struct LowerDenseMatrixPass
    : public mlir::PassWrapper<LowerDenseMatrixPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
    explicit LowerDenseMatrixPass() {}

    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<mlir::LLVM::LLVMDialect, mlir::AffineDialect, mlir::memref::MemRefDialect>();
    }
    void runOnOperation() final;
};
}  // end anonymous namespace

void LowerDenseMatrixPass::runOnOperation() {
    mlir::ConversionTarget target(getContext());
    mlir::RewritePatternSet patterns(&getContext());
    LowerToLLVMOptions llvmOptions(&getContext());
    LLVMTypeConverter typeConverter(&getContext(), llvmOptions);

    target.addLegalDialect<mlir::memref::MemRefDialect>();
    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addLegalDialect<mlir::scf::SCFDialect>();
    target.addLegalDialect<mlir::AffineDialect>();
    target.addLegalDialect<mlir::linalg::LinalgDialect>();
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();

    target.addLegalOp<mlir::daphne::GetMemRefDenseMatrix>();
    target.addLegalOp<mlir::daphne::GetDenseMatrixFromMemRef>();
    target.addLegalOp<mlir::daphne::PrintMemRef>();
    target.addIllegalOp<mlir::daphne::AllAggSumOp>();
    target.addIllegalOp<mlir::daphne::MatMulOp>();
    // populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);

    typeConverter.addConversion([&](daphne::MatrixType t) {
        return mlir::MemRefType::get({t.getNumRows(), t.getNumCols()},
                                     t.getElementType());
    });

    patterns.insert<MatMulOpLowering, SumAllOpLowering>(&getContext());
    auto module = getOperation();
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

class MemRefCallingConvention : public OpConversionPattern<daphne::GetDenseMatrixFromMemRef> {
   public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        daphne::GetDenseMatrixFromMemRef op, OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override {

        /*
        std::cout <<"hugo\n";
        auto loc = op->getLoc();
        // SmallVector<mlir::Value> callOperands{};
        // auto extractStridedMetadataOp =
        //     rewriter.create<memref::ExtractStridedMetadataOp>(loc, op.getArg());
        // // Base ptr.
        // callOperands.push_back(extractStridedMetadataOp.getBaseBuffer());
        // // offset.
        // callOperands.push_back(extractStridedMetadataOp.getOffset());
        // // strides.
        // callOperands.push_back(extractStridedMetadataOp.getStrides().front());

        // std::string callee_name = "_getDenseMatrixFromMemRef__DenseMatrix_double__StridedMemRefType_"
        std::string fName = "_printMemRef"; 
        mlir::Value memRef = adaptor.getArg();
        ArrayRef<mlir::Value> args({memRef});

        TypeRange ts;
        auto callOp = rewriter.create<func::CallOp>(loc, fName, ts, args);
        // rewriter.replaceOp(op, memRef);
        // SmallVector<Value, 4> results;
        // results.append(callOp.result_begin(), callOp.result_end());
        rewriter.replaceOp(op, callOp.getResult(0));
        */
        return success();
    }
};

namespace {
struct MemRefCallingConventionPass
    : public mlir::PassWrapper<MemRefCallingConventionPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
    explicit MemRefCallingConventionPass() {}

    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<mlir::daphne::DaphneDialect, mlir::LLVM::LLVMDialect,
                        mlir::AffineDialect, mlir::memref::MemRefDialect>();
    }
    void runOnOperation() final;
};
}  // namespace

void MemRefCallingConventionPass::runOnOperation() {
    mlir::ConversionTarget target(getContext());
    mlir::RewritePatternSet patterns(&getContext());
    LowerToLLVMOptions llvmOptions(&getContext());
    // llvmOptions.seBarePtrCallConv = true;
    LLVMTypeConverter typeConverter(&getContext(), llvmOptions);
    typeConverter.addConversion([&](daphne::MatrixType t) {
        return mlir::MemRefType::get({t.getNumRows(), t.getNumCols()},
                                     t.getElementType());
    });

    populateAffineToStdConversionPatterns(patterns);
    populateSCFToControlFlowConversionPatterns(patterns);
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);

    target.addLegalDialect<mlir::memref::MemRefDialect>();
    target.addLegalDialect<mlir::func::FuncDialect>();
    // target.addLegalDialect<mlir::arith::ArithDialect>();
    // target.addLegalDialect<mlir::scf::SCFDialect>();
    // target.addLegalDialect<mlir::AffineDialect>();
    // target.addLegalDialect<mlir::linalg::LinalgDialect>();
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    // populate

    target.addIllegalOp<mlir::daphne::GetDenseMatrixFromMemRef>();

    patterns.insert<MemRefCallingConvention>(&getContext());
    auto module = getOperation();
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<mlir::Pass> mlir::daphne::createLowerDenseMatrixPass() {
    return std::make_unique<LowerDenseMatrixPass>();
}

std::unique_ptr<mlir::Pass> mlir::daphne::createMemRefTestPass()
{
    return std::make_unique<MemRefCallingConventionPass>();
}
