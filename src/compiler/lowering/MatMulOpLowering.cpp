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

#include <cstdint>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "compiler/utils/CompilerUtils.h"
#include "compiler/utils/LoweringUtils.h"
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
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
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

static constexpr int ROW = 0;
static constexpr int COL = 1;

llvm::SmallVector<AffineForOp, 3> affineMatMul(mlir::Value &lhs, mlir::Value &rhs, mlir::Value &output,
                  ConversionPatternRewriter &rewriter, mlir::Location loc,
                  ArrayRef<int64_t> lhsShape, ArrayRef<int64_t> rhsShape,
                  mlir::MLIRContext *ctx, Type elementType, int64_t vec_size) {
    llvm::SmallVector<AffineForOp, 3> loops;
    int64_t last_vec_size = rhsShape[COL] % vec_size;
    auto vec_Type = mlir::VectorType::get({vec_size}, elementType);
        
    // TODO: We need an option to enable smaller vector sizes for the ends of each row.
    // row loop
    auto rowLoop = rewriter.create<AffineForOp>(loc, 0, lhsShape[ROW], 1);
    // row loop body
    rewriter.setInsertionPointToStart(rowLoop.getBody());
    // col loop
    auto colLoop = rewriter.create<AffineForOp>(loc, 0, rhsShape[COL] - last_vec_size, vec_size);
    // col loop body
    rewriter.setInsertionPointToStart(colLoop.getBody());
    // fma loop
    auto fmaLoop = rewriter.create<AffineForOp>(loc, 0, rhsShape[ROW], 1);
    // inner loop body
    rewriter.setInsertionPointToStart(fmaLoop.getBody());
    
        auto a_single = rewriter.create<AffineLoadOp>(loc, lhs, 
                                                    ValueRange{rowLoop.getInductionVar(), fmaLoop.getInductionVar()});
        auto  a = rewriter.create<vector::SplatOp>(loc, a_single, vec_Type);
        auto  b = rewriter.create<AffineVectorLoadOp>(
                   loc, vec_Type, rhs,
                    ValueRange{fmaLoop.getInductionVar(), colLoop.getInductionVar()});
        auto  c = rewriter.create<AffineVectorLoadOp>(
                    loc, vec_Type, output,
                    ValueRange{rowLoop.getInductionVar(), colLoop.getInductionVar()});
        
        Value res = rewriter.create<vector::FMAOp>(loc, a, b, c);
       
        rewriter.create<AffineVectorStoreOp>(loc, res, output,
                                           ValueRange{rowLoop.getInductionVar(), colLoop.getInductionVar()});


    
    // AffineYieldOp at end of loop blocks
    rewriter.setInsertionPointAfter(fmaLoop);
    rewriter.setInsertionPointAfter(colLoop);
    rewriter.setInsertionPointAfter(rowLoop);

    loops.push_back(rowLoop);
    loops.push_back(colLoop);
    loops.push_back(fmaLoop);
    return loops;
}

// Simple asserts in the matchAndRewrite don't have any effect
void print_assert(bool statement, std::string s) {
    if (!statement) {
        std::cout << s << std::endl;
    }
}


class MatMulLowering : public OpConversionPattern<daphne::MatMulOp> {
   public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        daphne::MatMulOp op, OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override {
        int64_t vec_size = 4;
        auto loc = op->getLoc();
        mlir::daphne::MatrixType lhsMatrixType =
            adaptor.getLhs().getType().dyn_cast<mlir::daphne::MatrixType>();
        mlir::daphne::MatrixType rhsMatrixType =
            adaptor.getRhs().getType().dyn_cast<mlir::daphne::MatrixType>();

        auto lhsRows = lhsMatrixType.getNumRows();
        auto lhsCols = lhsMatrixType.getNumCols();

        auto rhsRows = rhsMatrixType.getNumRows();
        auto rhsCols = rhsMatrixType.getNumCols();

        auto matrixElementType = lhsMatrixType.getElementType();

        // TODO(phil): if shape is unknown, e.g., row/col = -1 we currently
        // can't create a MemRefType
        auto lhsMemRefType =
            mlir::MemRefType::get({lhsRows, lhsCols}, matrixElementType);
        auto rhsMemRefType =
            mlir::MemRefType::get({rhsRows, rhsCols}, matrixElementType);

        mlir::MemRefType outputMemRefType =
            mlir::MemRefType::get({lhsRows, rhsCols}, matrixElementType);

        // daphne::Matrix -> memref
        mlir::Value lhs =
            rewriter.create<mlir::daphne::ConvertDenseMatrixToMemRef>(
                op->getLoc(), lhsMemRefType, adaptor.getLhs());
        mlir::Value rhs =
            rewriter.create<mlir::daphne::ConvertDenseMatrixToMemRef>(
                op->getLoc(), rhsMemRefType, adaptor.getRhs());

        // Alloc output memref
        mlir::Value outputMemRef =
            insertMemRefAlloc(outputMemRefType, loc, rewriter);

        // Fill the output MemRef
        affineFillMemRef(0.0, rewriter, loc, outputMemRefType.getShape(),
                         op->getContext(), outputMemRef, matrixElementType);
        // Do the actual MatMul with hand built codegen
        auto loops = affineMatMul(lhs, rhs, outputMemRef, rewriter, loc,
                     lhsMemRefType.getShape(), rhsMemRefType.getShape(),
                     op->getContext(), matrixElementType, vec_size);

        unsigned MC = 64;
        unsigned KC = 256;
        unsigned NC = 512;
        unsigned NR = std::max((int64_t)1, 4 / vec_size);
        unsigned MR = 4;
        unsigned KU = 4;
        llvm::SmallVector<AffineForOp> loopNest;
        getPerfectlyNestedLoops(loopNest, loops.front());
        // tile i with MC, j with NC, k with KC
        llvm::SmallVector<AffineForOp> tiledNest;
        if (failed(tilePerfectlyNested(loopNest, {MC, NC, KC}, &tiledNest))) {
            std::cout << "Failed to tile the Loop nest" << std::endl;
        };
        print_assert(tiledNest[0].getStep() == MC, "0 should have step size MC.");
        print_assert(tiledNest[1].getStep() == NC * vec_size, "1 should have step size NC * vec_size.");
        print_assert(tiledNest[2].getStep() == KC, "2 should have step size KC.");
        print_assert(tiledNest[3].getStep() == 1, "3 should have step size 1.");
        print_assert(tiledNest[4].getStep() == 1 * vec_size, "4 should have step size vec_size.");
        print_assert(tiledNest[5].getStep() == 1, "5 should have step size 1.");

        // Further tile the i mod MC loop with MR
        if (failed(tilePerfectlyNested(tiledNest[3], {MR}))) {
            std::cout << "Failed to tile the second Loop nest" << std::endl;
        };
        
        // Further tile the j mod NC loop with NR
        print_assert(tiledNest[4].getStep() == 1 * vec_size, "4 should have step size vec_size.");
        if (failed(tilePerfectlyNested(tiledNest[4], {NR}))) {
            std::cout << "Failed to tile the second j Loop" << std::endl;
        };


        llvm::SmallVector<AffineForOp> twiceTiledNest;
        getPerfectlyNestedLoops(twiceTiledNest, tiledNest[0]);
        print_assert(twiceTiledNest[0].getStep() == MC, "tTN: 0 should have step size MC.");  // i loops
        print_assert(twiceTiledNest[3].getStep() == MR, "tTN: 3 should have step size MR.");
        print_assert(twiceTiledNest[4].getStep() == 1, "tTN: 4 should have step size 1.");
        print_assert(twiceTiledNest[1].getStep() == NC * vec_size, "tTN: 1 should have step size NC * vec_size.");  // j loops
        print_assert(twiceTiledNest[5].getStep() == NR * vec_size, "tTN: 5 should have step size NR.");
        print_assert(twiceTiledNest[6].getStep() == 1 * vec_size, "tTN: 6 should have step size vec_size.");
        print_assert(twiceTiledNest[2].getStep() == KC, "tTN: 2 should have step size 1.");  // k loops
        print_assert(twiceTiledNest[7].getStep() == 1, "tTN: 7 should have step size 1."); 

                                    
        // permute loops to final order (i / MC, j / NC, k / KC, i / MR, i mod MR, j / NR, j mod NR, k mod KC) ->
        //                              (j / NC, k / KC, i / MC, j / NR, i / MR, k mod KC, j mod NR, i mod MR)
        // TODO: This assert only fails in debug mode?!
        //assert(isValidLoopInterchangePermutation(twiceTiledNest, {2, 0, 1, 4, 7, 3, 6, 5}));
        unsigned root_idx = permuteLoops(twiceTiledNest, {2, 0, 1, 4, 7, 3, 6, 5});

        // Unroll and jam
        llvm::SmallVector<AffineForOp> blisTiledLoops;
        getPerfectlyNestedLoops(blisTiledLoops, twiceTiledNest[root_idx]); 
        print_assert(blisTiledLoops[2].getStep() == MC, "blisTiled: 2 should have step size MC.");  // i loops
        print_assert(blisTiledLoops[4].getStep() == MR, "blisTiled: 4 should have step size MR.");
        print_assert(blisTiledLoops[7].getStep() == 1, "blisTiled: 7 should have step size 1.");
        print_assert(blisTiledLoops[0].getStep() == NC * vec_size, "blisTiled: 0 should have step size NC * vec_size.");  // j loops
        print_assert(blisTiledLoops[3].getStep() == NR * vec_size, "blisTiled: 3 should have step size NR * vec_size.");
        print_assert(blisTiledLoops[6].getStep() == vec_size, "blisTiled: 6 should have step size vec_size.");
        print_assert(blisTiledLoops[1].getStep() == KC, "blisTiled: 1 should have step size 1.");  // k loops
        print_assert(blisTiledLoops[5].getStep() == 1, "blisTiled: 5 should have step size 1.");
        // TODO: This Matmul fails, if the last loops are not unrolled?
        if (failed(loopUnrollJamUpToFactor(blisTiledLoops[7], MR))) {
            std::cout << "Could not unroll the last loop" << std::endl;
        }
        if (failed(loopUnrollJamUpToFactor(blisTiledLoops[6], NR))) {
            std::cout << "Could not unroll the second to last loop" << std::endl;
        }
        if (failed(loopUnrollUpToFactor(blisTiledLoops[5], KU))) {
            std::cout << "Could not unroll the K loop" << std::endl;
        }
        
        mlir::Value DM = convertMemRefToDenseMatrix(loc, rewriter, outputMemRef,
                                                    op.getType());
        std::cout << "Converted back to Dense Matrix" << std::endl;
        rewriter.replaceOp(op, DM);
        return success();
    }
};


namespace {
/**
 * @brief The MatMulLoweringPass rewrites the MatMulOp from the DaphneDialect
 * to a affine loop structure implementing a multi tiled loop structure.
 *
 * The choice of tile sizes is taken from https://github.com/bondhugula/llvm-project/blob/hop/mlir/docs/HighPerfCodeGen.md
 */
struct MatMulLoweringPass
    : public mlir::PassWrapper<MatMulLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
    explicit MatMulLoweringPass() {}

    StringRef getArgument() const final { return "lower-mm"; }
    StringRef getDescription() const final {
        return "This pass lowers the MatMulOp to an affine loop structure.";
    }

    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<mlir::LLVM::LLVMDialect, mlir::AffineDialect,
                        mlir::memref::MemRefDialect>();
    }
    void runOnOperation() final;
};
}  // end anonymous namespace

void MatMulLoweringPass::runOnOperation() {
    auto module = getOperation();
    {
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
        target.addLegalDialect<mlir::vector::VectorDialect>();
        
        target.addLegalOp<mlir::daphne::ConvertDenseMatrixToMemRef>();
        target.addLegalOp<mlir::daphne::ConvertMemRefToDenseMatrix>();
        target.addLegalOp<mlir::daphne::DecRefOp>();

        target.addIllegalOp<mlir::daphne::MatMulOp>();

        patterns.insert<MatMulLowering>(&getContext());
        
        if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
}

std::unique_ptr<mlir::Pass> mlir::daphne::createMatMulOpLoweringPass() {
    return std::make_unique<MatMulLoweringPass>();
}
