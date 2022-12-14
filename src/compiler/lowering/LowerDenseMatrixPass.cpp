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
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APFloat.h"

using namespace mlir;

class PrintOpLowering : public OpConversionPattern<daphne::PrintOp> {
   public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        daphne::PrintOp op, ArrayRef<Value> operands,
        ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto nR = rewriter.create<ConstantIndexOp>(loc, 10);
    auto tensorType = Float64Type::get(op->getContext());
    auto memRefType = mlir::MemRefType::get(
        {nR, nR}, tensorType);
    auto memRefShape = memRefType.getShape();


    ModuleOp parentModule = op->getParentOfType<ModuleOp>();

    // Get a symbol reference to the printf function, inserting it if necessary.
    auto printfRef = getOrInsertPrintf(rewriter, parentModule);
    Value formatSpecifierCst = getOrCreateGlobalString(
        loc, rewriter, "frmt_spec", StringRef("%f \0", 4), parentModule);
    Value newLineCst = getOrCreateGlobalString(
        loc, rewriter, "nl", StringRef("\n\0", 2), parentModule);

    // Create a loop for each of the dimensions within the shape.
    SmallVector<Value, 4> loopIvs;
    for (unsigned i = 0, e = memRefShape.size(); i != e; ++i) {
      auto lowerBound = rewriter.create<ConstantIndexOp>(loc, 0);
      auto upperBound = rewriter.create<ConstantIndexOp>(loc, memRefShape[i]);
      auto step = rewriter.create<ConstantIndexOp>(loc, 1);
      auto loop =
          rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
      for (Operation &nested : *loop.getBody())
        rewriter.eraseOp(&nested);
      loopIvs.push_back(loop.getInductionVar());

      // Terminate the loop body.
      rewriter.setInsertionPointToEnd(loop.getBody());

      // Insert a newline after each of the inner dimensions of the shape.
      if (i != e - 1)
        rewriter.create<CallOp>(loc, printfRef, rewriter.getIntegerType(32),
                                newLineCst);
      rewriter.create<scf::YieldOp>(loc);
      rewriter.setInsertionPointToStart(loop.getBody());
    }

    // Generate a call to printf for the current element of the loop.
    // auto printOp = cast<daphne::PrintOp>(op);
    auto elementLoad =
        rewriter.create<memref::LoadOp>(loc, operands[0], loopIvs);
    rewriter.create<CallOp>(loc, printfRef, rewriter.getIntegerType(32),
                            ArrayRef<Value>({formatSpecifierCst, elementLoad}));

    // Notify the rewriter that this operation has been removed.
    rewriter.eraseOp(op);
    return success();
    }

private:
  /// Return a symbol reference to the printf function, inserting it into the
  /// module if necessary.
  static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                             ModuleOp module) {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
      return SymbolRefAttr::get(context, "printf");

    // Create a function declaration for printf, the signature is:
    //   * `i32 (i8*, ...)`
    auto llvmI32Ty = IntegerType::get(context, 32);
    auto llvmI8PtrTy = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
    auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI32Ty, llvmI8PtrTy,
                                                  /*isVarArg=*/true);

    // Insert the printf function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf", llvmFnType);
    return SymbolRefAttr::get(context, "printf");
  }

  /// Return a value representing an access into a global string with the given
  /// name, creating the string if necessary.
  static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                       StringRef name, StringRef value,
                                       ModuleOp module) {
    // Create the global at the entry of the module.
    LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
      OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = LLVM::LLVMArrayType::get(
          IntegerType::get(builder.getContext(), 8), value.size());
      global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                              LLVM::Linkage::Internal, name,
                                              builder.getStringAttr(value),
                                              0);
    }

    // Get the pointer to the first character in the global string.
    Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
    Value cst0 = builder.create<LLVM::ConstantOp>(
        loc, IntegerType::get(builder.getContext(), 64),
        builder.getIntegerAttr(builder.getIndexType(), 0));
    return builder.create<LLVM::GEPOp>(
        loc,
        LLVM::LLVMPointerType::get(IntegerType::get(builder.getContext(), 8)),
        globalPtr, ArrayRef<Value>({cst0, cst0}));
  }
};

class FillOpLowering : public OpConversionPattern<daphne::FillOp> {
   public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        daphne::FillOp op, ArrayRef<Value> operands,
        ConversionPatternRewriter &rewriter) const override {
        return success();
    }
};

class MatMulOpLowering : public OpConversionPattern<daphne::MatMulOp> {
   public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        daphne::MatMulOp op, ArrayRef<Value> operands,
        ConversionPatternRewriter &rewriter) const override {
        op.getType().dump();
        std::cout << "\n\n";

        auto loc = op->getLoc();
        mlir::daphne::MatrixType tensor =
            operands[0].getType().dyn_cast<mlir::daphne::MatrixType>();

        auto nR = tensor.getNumRows();
        auto nC = tensor.getNumCols();
        auto tensorType = tensor.getElementType();
        auto memRefType = mlir::MemRefType::get(
            {nR, nC}, tensorType);

        auto lhs = rewriter.create<mlir::daphne::GetMemRefDenseMatrix>(
            op->getLoc(), memRefType, operands[0]);
        auto rhs = rewriter.create<mlir::daphne::GetMemRefDenseMatrix>(
            op->getLoc(), memRefType, operands[1]);

        mlir::Value outputMemRef = rewriter.create<memref::AllocOp>(loc, memRefType);
        rewriter.create<linalg::MatmulOp>(loc, ValueRange{lhs, rhs}, ValueRange{outputMemRef});


        // llvm::APFloat zero = tensorType.isF32() ? llvm::APFloat(float(0)) : llvm::APFloat(0.0);
        // Value sum = rewriter.create<mlir::ConstantFloatOp>(
        //     op->getLoc(), zero, tensorType.dyn_cast<mlir::FloatType>());
        //
        // SmallVector<Value, 4> loopIvs;
        // // SmallVector<scf::ForOp, 2> forOps;
        // SmallVector<AffineForOp, 2> forOps;
        // // auto lowerBound = rewriter.create<ConstantIndexOp>(loc, 0);
        // // auto outerUpperBound =
        // //     rewriter.create<ConstantIndexOp>(loc, memRefShape[0]);
        // // auto step = rewriter.create<ConstantIndexOp>(loc, 1);
        // // outer loop
        // // auto outerLoop = rewriter.create<scf::ForOp>(
        // auto outerLoop = rewriter.create<AffineForOp>(
        //     loc, 0, nR, 1, ValueRange{sum});
        // for (Operation &nested : *outerLoop.getBody()) {
        //     rewriter.eraseOp(&nested);
        // }
        // loopIvs.push_back(outerLoop.getInductionVar());
        // // outer loop body
        // rewriter.setInsertionPointToStart(outerLoop.getBody());
        // Value sum_iter = rewriter.create<mlir::ConstantFloatOp>(
        //     op->getLoc(), zero,
        //     tensorType.dyn_cast<mlir::FloatType>());
        // // inner loop
        // // auto innerUpperBound =
        // //     rewriter.create<ConstantIndexOp>(loc, memRefShape[1]);
        // // auto innerLoop = rewriter.create<scf::ForOp>(
        // auto innerLoop = rewriter.create<AffineForOp>(
        //     loc, 0, nC, 1, ValueRange{sum_iter});
        // for (Operation &nested : *innerLoop.getBody()) {
        //     rewriter.eraseOp(&nested);
        // }
        // loopIvs.push_back(innerLoop.getInductionVar());
        // // inner loop body
        // rewriter.setInsertionPointToStart(innerLoop.getBody());
        // // load value from memref
        // auto elementLoad =
        //     rewriter.create<memref::LoadOp>(loc, outputMemRef, loopIvs);
        // // sum loop iter arg and memref value
        // mlir::Value inner_sum = rewriter.create<AddFOp>(
        //     loc, innerLoop.getRegionIterArgs()[0], elementLoad);
        // // yield inner loop result
        // rewriter.setInsertionPointToEnd(innerLoop.getBody());
        // // rewriter.create<scf::YieldOp>(loc, inner_sum);
        // rewriter.create<AffineYieldOp>(loc, inner_sum);
        // // yield outer loop result
        // rewriter.setInsertionPointToEnd(outerLoop.getBody());
        // mlir::Value outer_sum = rewriter.create<AddFOp>(
        //     loc, outerLoop.getRegionIterArgs()[0], innerLoop.getResult(0));
        // // rewriter.create<scf::YieldOp>(loc, outer_sum);
        // rewriter.create<AffineYieldOp>(loc, outer_sum);
        //
        // // replace sumAll op with result of loops
        // rewriter.replaceOp(op, outerLoop.getResult(0));
        rewriter.replaceOp(op, outputMemRef);
        return success();
    }
};

class SumAllOpLowering : public OpConversionPattern<daphne::AllAggSumOp> {
   public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        daphne::AllAggSumOp op, ArrayRef<Value> operands,
        ConversionPatternRewriter &rewriter) const override {
        mlir::daphne::MatrixType tensor =
            operands[0].getType().dyn_cast<mlir::daphne::MatrixType>();

        auto loc = op->getLoc();
        auto nR = tensor.getNumRows();
        auto nC = tensor.getNumCols();

        auto tensorType = tensor.getElementType();
        auto memRefType = mlir::MemRefType::get(
            {nR, nC}, tensorType);
        auto memRef = rewriter.create<mlir::daphne::GetMemRefDenseMatrix>(
            op->getLoc(), memRefType, operands[0]);

        llvm::APFloat zero = tensorType.isF32() ? llvm::APFloat(float(0)) : llvm::APFloat(0.0);
        Value sum = rewriter.create<mlir::ConstantFloatOp>(
            op->getLoc(), zero, tensorType.dyn_cast<mlir::FloatType>());

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
        Value sum_iter = rewriter.create<mlir::ConstantFloatOp>(
            op->getLoc(), zero,
            tensorType.dyn_cast<mlir::FloatType>());
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
        mlir::Value inner_sum = rewriter.create<AddFOp>(
            loc, innerLoop.getRegionIterArgs()[0], elementLoad);
        // yield inner loop result
        rewriter.setInsertionPointToEnd(innerLoop.getBody());
        // rewriter.create<scf::YieldOp>(loc, inner_sum);
        rewriter.create<AffineYieldOp>(loc, inner_sum);
        // yield outer loop result
        rewriter.setInsertionPointToEnd(outerLoop.getBody());
        mlir::Value outer_sum = rewriter.create<AddFOp>(
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
        registry.insert<mlir::LLVM::LLVMDialect>();
    }
    void runOnOperation() final;
};
}  // end anonymous namespace

void LowerDenseMatrixPass::runOnOperation() {
    mlir::ConversionTarget target(getContext());
    mlir::OwningRewritePatternList patterns(&getContext());
    LowerToLLVMOptions llvmOptions(&getContext());
    llvmOptions.emitCWrappers = true;
    LLVMTypeConverter typeConverter(&getContext(), llvmOptions);

    target.addLegalDialect<mlir::memref::MemRefDialect>();
    target.addLegalDialect<mlir::StandardOpsDialect>();
    target.addLegalDialect<mlir::scf::SCFDialect>();
    target.addLegalDialect<mlir::AffineDialect>();
    target.addLegalDialect<mlir::linalg::LinalgDialect>();
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();

    target.addLegalOp<mlir::daphne::GetMemRefDenseMatrix>();
    target.addIllegalOp<mlir::daphne::AllAggSumOp>();

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

std::unique_ptr<mlir::Pass> mlir::daphne::createLowerDenseMatrixPass() {
    return std::make_unique<LowerDenseMatrixPass>();
}
