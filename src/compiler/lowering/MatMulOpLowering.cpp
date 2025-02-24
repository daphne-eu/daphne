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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "compiler/utils/LoweringUtils.h"
#include "hwloc.h"
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
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/Utils.h"
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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "spdlog/spdlog.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <util/ErrorHandler.h>

namespace mlir {
#define GEN_PASS_DECL_MATMULOPLOWERINGPASS
#define GEN_PASS_DEF_MATMULOPLOWERINGPASS
#include "ir/daphneir/Passes.h.inc"
} // namespace mlir

using namespace mlir;

static constexpr int ROW = 0;
static constexpr int COL = 1;

struct LowerMatMulOpOptions {
    LowerMatMulOpOptions() {}
    int vec_size_bits{0};
    int num_vec_registers{0};
    bool vectorize{false};
    bool tile{false};
    bool invert_loops{false};
    bool useFixedTileSizes{false};
    llvm::SmallVector<int, 3> cache_sizes;
    llvm::SmallVector<unsigned, 5> tile_sizes;
    int unroll_factor{0};
    int unroll_jam_factor{0};

    LowerMatMulOpOptions &setTileSizes(std::vector<unsigned> sizes) {
        tile_sizes.clear();
        for (auto s : sizes) {
            tile_sizes.push_back(s);
        }
        return *this;
    }
    LowerMatMulOpOptions &setUnrollFactor(int f) {
        unroll_factor = f;
        return *this;
    }
    LowerMatMulOpOptions &setUnrollJamFactor(int f) {
        unroll_jam_factor = f;
        return *this;
    }
    LowerMatMulOpOptions &setCacheSizes(llvm::SmallVector<int> caches) {
        cache_sizes.clear();
        for (auto c : caches) {
            cache_sizes.push_back(c);
        }
        return *this;
    }
    LowerMatMulOpOptions &enableVectorization(bool b = true) {
        vectorize = b;
        return *this;
    }
    LowerMatMulOpOptions &setVectorSizeBits(int s) {
        vec_size_bits = s;
        return *this;
    }
    LowerMatMulOpOptions &setNumberOfVectorRegisters(int s) {
        num_vec_registers = s;
        return *this;
    }
    LowerMatMulOpOptions &enableTiling(bool b = true) {
        tile = b;
        return *this;
    }
    LowerMatMulOpOptions &enableLoopInversion(bool b = true) {
        invert_loops = b;
        return *this;
    }
    int getVecSize(int bitwidth) const {
        if (vec_size_bits > 0) {
            return std::max(1, vec_size_bits / bitwidth);
        } else {
            return 1;
        }
    }
    int getRegisterSize() const {
        if (num_vec_registers != 0 && vec_size_bits != 0) {
            return std::max(1, num_vec_registers * vec_size_bits);
        }
        return 1;
    }
};

bool is_valid_options(LowerMatMulOpOptions const options) {
    for (auto s : options.tile_sizes)
        if (s <= 1) {
            spdlog::warn("Tile sizes must be an integer larger than 1.");
            return false;
        }
    if (options.unroll_factor < 0) {
        spdlog::warn("Unroll factor must be an integer >= 0.");
        return false;
    }
    if (options.unroll_jam_factor < 0) {
        spdlog::warn("Unroll jam factor must be an integer >= 0.");
        return false;
    }
    if (options.vec_size_bits < 0) {
        spdlog::warn("Vector size bits must be an integer >= 0.");
        return false;
    }
    return true;
}

class MatMulLowering : public OpConversionPattern<daphne::MatMulOp> {
    const LowerMatMulOpOptions options;

  public:
    using OpConversionPattern::OpConversionPattern;
    explicit MatMulLowering(mlir::TypeConverter &typeConverter, MLIRContext *context,
                            LowerMatMulOpOptions const &options)
        : OpConversionPattern<daphne::MatMulOp>(typeConverter, context, PatternBenefit(1)), options(options) {
        this->setDebugName("MatMulLowering");
    }

    bool is_vectorizable(ArrayRef<int64_t> const rhsShape, Type const matrixElementType) const {
        if (rhsShape[COL] % options.getVecSize(matrixElementType.getIntOrFloatBitWidth()) != 0) {
            return false;
        }
        if (!matrixElementType.isa<FloatType>()) {
            return false;
        }
        return true;
    }

    bool is_tileable(ArrayRef<int64_t> const rhsShape) const { return true; }

    llvm::SmallVector<AffineForOp, 3> affineMatMul(mlir::Value &lhs, mlir::Value &rhs, mlir::Value &output,
                                                   ConversionPatternRewriter &rewriter, mlir::Location loc,
                                                   ArrayRef<int64_t> lhsShape, ArrayRef<int64_t> rhsShape,
                                                   mlir::MLIRContext *ctx, SmallVector<AffineForOp, 3> &loops,
                                                   Type elementType) const {
        // row loop
        auto rowLoop = rewriter.create<AffineForOp>(loc, 0, lhsShape[ROW], 1);
        // row loop body
        rewriter.setInsertionPointToStart(rowLoop.getBody());
        // col loop
        auto colLoop = rewriter.create<AffineForOp>(loc, 0, rhsShape[COL], 1);
        // col loop body
        rewriter.setInsertionPointToStart(colLoop.getBody());
        // fma loop
        auto fmaLoop = rewriter.create<AffineForOp>(loc, 0, rhsShape[ROW], 1);
        // inner loop body
        rewriter.setInsertionPointToStart(fmaLoop.getBody());

        auto a =
            rewriter.create<AffineLoadOp>(loc, lhs, ValueRange{rowLoop.getInductionVar(), fmaLoop.getInductionVar()});
        auto b =
            rewriter.create<AffineLoadOp>(loc, rhs, ValueRange{fmaLoop.getInductionVar(), colLoop.getInductionVar()});
        auto c = rewriter.create<AffineLoadOp>(loc, output,
                                               ValueRange{rowLoop.getInductionVar(), colLoop.getInductionVar()});
        if (elementType.isIntOrIndex()) {
            // Arith operates on MLIR signless integers, while Daphne uses
            // (un)signed integers.
            Value castedA = this->typeConverter->materializeTargetConversion(
                rewriter, loc, rewriter.getIntegerType(elementType.getIntOrFloatBitWidth()), ValueRange{a});
            Value castedB = this->typeConverter->materializeTargetConversion(
                rewriter, loc, rewriter.getIntegerType(elementType.getIntOrFloatBitWidth()), ValueRange{b});
            Value castedC = this->typeConverter->materializeTargetConversion(
                rewriter, loc, rewriter.getIntegerType(elementType.getIntOrFloatBitWidth()), ValueRange{c});
            Value added = rewriter.create<arith::MulIOp>(loc, castedA, castedB);
            Value res = rewriter.create<arith::AddIOp>(loc, added, castedC);
            Value castedRes =
                this->typeConverter->materializeSourceConversion(rewriter, loc, elementType, ValueRange{res});
            rewriter.create<AffineStoreOp>(loc, castedRes, output,
                                           ValueRange{rowLoop.getInductionVar(), colLoop.getInductionVar()});
        } else {
            Value res = rewriter.create<LLVM::FMAOp>(loc, a, b, c);
            rewriter.create<AffineStoreOp>(loc, res, output,
                                           ValueRange{rowLoop.getInductionVar(), colLoop.getInductionVar()});
        }

        // AffineYieldOp at end of loop blocks
        rewriter.setInsertionPointAfter(fmaLoop);
        rewriter.setInsertionPointAfter(colLoop);
        rewriter.setInsertionPointAfter(rowLoop);

        loops.push_back(rowLoop);
        loops.push_back(colLoop);
        loops.push_back(fmaLoop);
        return loops;
    }

    llvm::SmallVector<AffineForOp, 3> vectorizedAffineMatMul(mlir::Value &lhs, mlir::Value &rhs, mlir::Value &output,
                                                             ConversionPatternRewriter &rewriter, mlir::Location loc,
                                                             ArrayRef<int64_t> lhsShape, ArrayRef<int64_t> rhsShape,
                                                             mlir::MLIRContext *ctx,
                                                             llvm::SmallVector<AffineForOp, 3> &loops, Type elementType,
                                                             int64_t vec_size) const {
        auto vec_Type = mlir::VectorType::get({vec_size}, elementType);

        // row loop
        auto rowLoop = rewriter.create<AffineForOp>(loc, 0, lhsShape[ROW], 1);
        // row loop body
        rewriter.setInsertionPointToStart(rowLoop.getBody());
        // col loop
        auto colLoop = rewriter.create<AffineForOp>(loc, 0, rhsShape[COL], vec_size);
        // col loop body
        rewriter.setInsertionPointToStart(colLoop.getBody());
        // fma loop
        auto fmaLoop = rewriter.create<AffineForOp>(loc, 0, rhsShape[ROW], 1);
        // inner loop body
        rewriter.setInsertionPointToStart(fmaLoop.getBody());

        auto a_single =
            rewriter.create<AffineLoadOp>(loc, lhs, ValueRange{rowLoop.getInductionVar(), fmaLoop.getInductionVar()});
        auto a = rewriter.create<vector::SplatOp>(loc, a_single, vec_Type);
        auto b = rewriter.create<AffineVectorLoadOp>(loc, vec_Type, rhs,
                                                     ValueRange{fmaLoop.getInductionVar(), colLoop.getInductionVar()});
        auto c = rewriter.create<AffineVectorLoadOp>(loc, vec_Type, output,
                                                     ValueRange{rowLoop.getInductionVar(), colLoop.getInductionVar()});

        // TODO: Integer doesn't actually work yet, so is disabled in
        // is_vectorizable.
        if (elementType.isIntOrIndex()) {
            Value added = rewriter.create<arith::MulIOp>(loc, a, b);
            Value res = rewriter.create<arith::AddIOp>(loc, added, c);
            rewriter.create<AffineVectorStoreOp>(loc, res, output,
                                                 ValueRange{rowLoop.getInductionVar(), colLoop.getInductionVar()});
        } else {
            Value res = rewriter.create<vector::FMAOp>(loc, a, b, c);
            rewriter.create<AffineVectorStoreOp>(loc, res, output,
                                                 ValueRange{rowLoop.getInductionVar(), colLoop.getInductionVar()});
        }

        // AffineYieldOp at end of loop blocks
        rewriter.setInsertionPointAfter(fmaLoop);
        rewriter.setInsertionPointAfter(colLoop);
        rewriter.setInsertionPointAfter(rowLoop);

        loops.push_back(rowLoop);
        loops.push_back(colLoop);
        loops.push_back(fmaLoop);
        return loops;
    }
    
    template <typename IOp, typename FOp>
    Value binaryWithConversionFunc(OpBuilder &rewriter, Location loc, TypeConverter *typeConverter, Value lhs, Value rhs) const {
        Type resType = lhs.getType();
        Value res{};
        if (llvm::isa<mlir::IntegerType>(resType)) {
            lhs = convertToSignlessInt(rewriter, loc, typeConverter, lhs, resType);
            rhs = convertToSignlessInt(rewriter, loc, typeConverter, rhs, resType);
            res = rewriter.create<IOp>(loc, lhs, rhs).getResult();
            res = typeConverter->materializeTargetConversion(rewriter, loc, resType, res);
        } else {
            res = rewriter.create<FOp>(loc, lhs, rhs).getResult();
        }
        return res;
    }

    template <arith::CmpIPredicate cmpIPredicate, arith::CmpIPredicate cmpFPredicate>
    Value cmpWithConversionFunc(OpBuilder &rewriter, Location loc, TypeConverter *typeConverter, Value lhs, Value rhs) const {
        Type resType = lhs.getType();
        Value res{};
        if (llvm::isa<mlir::IntegerType>(resType)) {
            lhs = convertToSignlessInt(rewriter, loc, typeConverter, lhs, resType);
            rhs = convertToSignlessInt(rewriter, loc, typeConverter, rhs, resType);
            res = rewriter.create<arith::CmpIOp>(loc, cmpIPredicate, lhs, rhs).getResult();
            res = typeConverter->materializeTargetConversion(rewriter, loc, resType, res);
        } else {
            res = rewriter.create<arith::CmpFOp>(loc, cmpFPredicate, lhs, rhs).getResult();
        }
        return res;
    }
    
    Value csrIndex(OpBuilder &rewriter, Location loc, 
        Value valuesMemRef, Value colIdxsMemRef, Value rowOffsetsMemRef, Value row, Value col, Type type) const 
    {
        auto zeroElem = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(type));
        auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
        auto rowPtr = row;
        auto nextRowPtr = rewriter.create<arith::AddIOp>(loc, row, one);
        auto colIdxLowerIncl = rewriter.create<memref::LoadOp>(
            loc, rowOffsetsMemRef, ValueRange{rowPtr});
        auto colIdxUpperExcl = rewriter.create<memref::LoadOp>(
            loc, rowOffsetsMemRef, ValueRange{nextRowPtr});
        auto search = rewriter.create<scf::ForOp>(
            loc, colIdxLowerIncl, colIdxUpperExcl, one, ValueRange{zeroElem},
            [&](OpBuilder &OpBuilderNested, Location locNested, Value loopIdx, ValueRange loopIterArgs)
            {
                auto getCol = OpBuilderNested.create<memref::LoadOp>(locNested, colIdxsMemRef, ValueRange{loopIdx});
                auto getValue = OpBuilderNested.create<memref::LoadOp>(locNested, valuesMemRef, ValueRange{loopIdx});
                auto cond = OpBuilderNested.create<arith::CmpIOp>(locNested, arith::CmpIPredicate::eq, getCol, col);
                // return the value of non-zero element if exists, else return a zero value
                auto res = OpBuilderNested.create<scf::IfOp>(
                    locNested, cond,
                    [&](OpBuilder &OpBuilderTwiceNested, Location locTwiceNested)
                    {
                        OpBuilderTwiceNested.create<scf::YieldOp>(locTwiceNested, ValueRange{getValue});
                    },
                    [&](OpBuilder &OpBuilderTwiceNested, Location locTwiceNested)
                    {
                        OpBuilderTwiceNested.create<scf::YieldOp>(locTwiceNested, ValueRange{zeroElem});
                    }
                );
                OpBuilderNested.create<scf::YieldOp>(locNested, res.getResult(0));
            }
        );
        return search.getResult(0);
    }
    
    LogicalResult matchAndRewriteSparseSparseMat(daphne::MatMulOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
        Location loc = op->getLoc();
        Value lhs = adaptor.getLhs();
        Value rhs = adaptor.getRhs();

        mlir::daphne::MatrixType lhsMatrixType = lhs.getType().dyn_cast<mlir::daphne::MatrixType>();
        mlir::daphne::MatrixType rhsMatrixType = rhs.getType().dyn_cast<mlir::daphne::MatrixType>();

        auto lhsRows = lhsMatrixType.getNumRows();
        auto lhsCols = lhsMatrixType.getNumCols();

        auto rhsRows = rhsMatrixType.getNumRows();
        auto rhsCols = rhsMatrixType.getNumCols();

        auto matrixElementType = lhsMatrixType.getElementType();

        auto lhsValuesMemRefType =
            MemRefType::get({ShapedType::kDynamic}, matrixElementType);
        auto lhsColIdxsMemRefType = 
            MemRefType::get({ShapedType::kDynamic}, rewriter.getIndexType());
        auto lhsRowOffsetsMemRefType = 
            MemRefType::get({lhsRows + 1}, rewriter.getIndexType());
        auto rhsValuesMemRefType =
            MemRefType::get({ShapedType::kDynamic}, matrixElementType);
        auto rhsColIdxsMemRefType = 
            MemRefType::get({ShapedType::kDynamic}, rewriter.getIndexType());
        auto rhsRowOffsetsMemRefType = 
            MemRefType::get({rhsRows + 1}, rewriter.getIndexType());
        auto resValuesMemRefType =
            MemRefType::get({ShapedType::kDynamic}, matrixElementType);
        auto resColIdxsMemRefType = 
            MemRefType::get({ShapedType::kDynamic}, rewriter.getIndexType());
        auto resRowOffsetsMemRefType = 
            MemRefType::get({lhsRows + 1}, rewriter.getIndexType());

        auto lhsValuesMemRef =
            rewriter.create<daphne::ConvertCSRMatrixToValuesMemRef>(loc, lhsValuesMemRefType, lhs);
        auto lhsColIdxsMemRef =
            rewriter.create<daphne::ConvertCSRMatrixToColIdxsMemRef>(loc, lhsColIdxsMemRefType, lhs);
        auto lhsRowOffsetsMemRef =
            rewriter.create<daphne::ConvertCSRMatrixToRowOffsetsMemRef>(loc, lhsRowOffsetsMemRefType, lhs);
        auto rhsValuesMemRef =
            rewriter.create<daphne::ConvertCSRMatrixToValuesMemRef>(loc, lhsValuesMemRefType, rhs);
        auto rhsColIdxsMemRef =
            rewriter.create<daphne::ConvertCSRMatrixToColIdxsMemRef>(loc, lhsColIdxsMemRefType, rhs);
        auto rhsRowOffsetsMemRef =
            rewriter.create<daphne::ConvertCSRMatrixToRowOffsetsMemRef>(loc, lhsRowOffsetsMemRefType, rhs);
            
        auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
        auto zeroElement = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(matrixElementType));
        auto numLhsRowsValue = rewriter.create<arith::ConstantIndexOp>(loc, lhsRows);
        auto numRhsRowsValue = rewriter.create<arith::ConstantIndexOp>(loc, rhsRows);
        auto numLhsColsValue = rewriter.create<arith::ConstantIndexOp>(loc, lhsCols);
        auto numRhsColsValue = rewriter.create<arith::ConstantIndexOp>(loc, rhsCols);
    
        auto resValuesMemRef = rewriter.create<memref::AllocOp>(loc, resValuesMemRefType, ValueRange{one});
        auto resColIdxsMemRef = rewriter.create<memref::AllocOp>(loc, resColIdxsMemRefType, ValueRange{one});
        auto resRowOffsetsMemRef = rewriter.create<memref::AllocOp>(loc, resRowOffsetsMemRefType);
        rewriter.create<memref::StoreOp>(loc, zero, resRowOffsetsMemRef, ValueRange{zero});

        auto lhsRowLoop = rewriter.create<scf::ForOp>(
            loc, zero, numLhsRowsValue, one, ValueRange{zero},
            [&](OpBuilder &OpBuilderNested, Location locNested, Value loopIdx, ValueRange loopIterArgs) 
            {
                auto rowPtr = loopIdx;
                auto lhsRow = loopIdx;
                auto nextRowPtr = OpBuilderNested.create<arith::AddIOp>(locNested, loopIdx, one);
                auto resValuesPtr = loopIterArgs[0];

                auto lhsColIdxLowerIncl = OpBuilderNested.create<memref::LoadOp>(
                    locNested, lhsRowOffsetsMemRef, ValueRange{rowPtr});
                auto lhsColIdxUpperExcl = OpBuilderNested.create<memref::LoadOp>(
                    locNested, lhsRowOffsetsMemRef, ValueRange{nextRowPtr});

                auto rhsColLoop = OpBuilderNested.create<scf::ForOp>(
                    locNested, zero, numRhsColsValue, one, ValueRange{resValuesPtr},
                    [&](OpBuilder &OpBuilderTwiceNested, Location locTwiceNested, Value loopIdx, ValueRange loopIterArgs) 
                    {
                        auto resValuesPtr = loopIterArgs[0];
                        auto rhsCol = loopIdx;
                        auto lhsColLoop = OpBuilderTwiceNested.create<scf::ForOp>(
                            locTwiceNested, lhsColIdxLowerIncl, lhsColIdxUpperExcl, one, ValueRange{zeroElement},
                            [&](OpBuilder &OpBuilderThreetimesNested, Location locThreetimesNested, Value loopIdx, ValueRange loopIterArgs)
                            {
                                auto acc = loopIterArgs[0];
                                    
                                auto lhsElemRow = lhsRow;
                                auto lhsElemCol = OpBuilderThreetimesNested.create<memref::LoadOp>(
                                    locThreetimesNested, lhsColIdxsMemRef, ValueRange{loopIdx});
                                auto lhsElemValue = OpBuilderThreetimesNested.create<memref::LoadOp>(
                                    locThreetimesNested, lhsValuesMemRef, ValueRange{loopIdx});

                                auto rhsElemRow = lhsElemCol;
                                auto rhsElemCol = rhsCol;
                                // locate the required element in rhs corresponding to the lhs element
                                auto rhsElemValue = csrIndex(
                                    OpBuilderThreetimesNested, locThreetimesNested, 
                                    rhsValuesMemRef, rhsColIdxsMemRef, rhsRowOffsetsMemRef, 
                                    rhsElemRow, rhsElemCol, matrixElementType); 
                                
                                auto product = binaryWithConversionFunc<arith::MulIOp, arith::MulFOp>(
                                    OpBuilderThreetimesNested, locThreetimesNested, this->typeConverter, lhsElemValue, rhsElemValue);
                                auto newAcc = binaryWithConversionFunc<arith::AddIOp, arith::AddFOp>(
                                    OpBuilderThreetimesNested, locThreetimesNested, this->typeConverter, product, acc);

                                OpBuilderThreetimesNested.create<scf::YieldOp>(locThreetimesNested, ValueRange{newAcc});

                            }
                        );
                        auto cond = OpBuilderTwiceNested.create<arith::CmpFOp>(
                            locTwiceNested, arith::CmpFPredicate::OEQ, lhsColLoop.getResult(0), zeroElement);
                        // store the result if it is not zero
                        auto newPtr = OpBuilderTwiceNested.create<scf::IfOp>(
                            locTwiceNested, cond, 
                            [&](OpBuilder &OpBuilderThreetimesNested, Location locThreetimesNested)
                            {
                                auto newResValuesPtr = resValuesPtr;
                                OpBuilderThreetimesNested.create<scf::YieldOp>(locThreetimesNested, ValueRange{newResValuesPtr});        
                            },
                            [&](OpBuilder &OpBuilderThreetimesNested, Location locThreetimesNested)
                            {   
                                auto newResValuesPtr = OpBuilderThreetimesNested.create<arith::AddIOp>(
                                    locThreetimesNested, resValuesPtr, one);
                                OpBuilderThreetimesNested.create<memref::StoreOp>(
                                    locThreetimesNested, lhsColLoop.getResult(0), resValuesMemRef, ValueRange{resValuesPtr});
                                OpBuilderThreetimesNested.create<memref::StoreOp>(
                                    locThreetimesNested, rhsCol, resColIdxsMemRef, ValueRange{resValuesPtr});
                                OpBuilderThreetimesNested.create<scf::YieldOp>(
                                    locThreetimesNested, ValueRange{newResValuesPtr});
                            }
                        );
                        OpBuilderTwiceNested.create<scf::YieldOp>(locTwiceNested, ValueRange{newPtr.getResult(0)});
                    }
                );
                auto newResValuesPtr = rhsColLoop.getResult(0);   
                
                OpBuilderNested.create<memref::StoreOp>(
                    locNested, 
                    newResValuesPtr,
                    resRowOffsetsMemRef,
                    ValueRange{nextRowPtr});
                OpBuilderNested.create<scf::YieldOp>(locNested, ValueRange{newResValuesPtr});
            }
        );
        
        Value maxNumRowsValue = rewriter.create<arith::ConstantIndexOp>(loc, lhsRows);
        Value numColsValue = rewriter.create<arith::ConstantIndexOp>(loc, rhsCols);
        Value maxNumNonZerosValue = rewriter.create<arith::ConstantIndexOp>(loc, lhsRows * rhsCols);

        Value resCSRMatrix = convertMemRefToCSRMatrix(loc, rewriter,
            resValuesMemRef, resColIdxsMemRef, resRowOffsetsMemRef, 
            maxNumRowsValue, numColsValue, maxNumNonZerosValue, op.getType()); 

        if (!resCSRMatrix) {
            llvm::errs() << "Error: resCSRMatrix is null!\n";
        }
        
        rewriter.replaceOp(op, resCSRMatrix);

        return mlir::success();   
    }
    
    LogicalResult matchAndRewriteSparseDenseMat(daphne::MatMulOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
        Location loc = op->getLoc();
        Value lhs = adaptor.getLhs();
        Value rhs = adaptor.getRhs();

        mlir::daphne::MatrixType lhsMatrixType = lhs.getType().dyn_cast<mlir::daphne::MatrixType>();
        mlir::daphne::MatrixType rhsMatrixType = rhs.getType().dyn_cast<mlir::daphne::MatrixType>();

        auto lhsRows = lhsMatrixType.getNumRows();
        auto lhsCols = lhsMatrixType.getNumCols();

        auto rhsRows = rhsMatrixType.getNumRows();
        auto rhsCols = rhsMatrixType.getNumCols();

        auto matrixElementType = lhsMatrixType.getElementType();

        auto lhsValuesMemRefType =
            MemRefType::get({ShapedType::kDynamic}, matrixElementType);
        auto lhsColIdxsMemRefType = 
            MemRefType::get({ShapedType::kDynamic}, rewriter.getIndexType());
        auto lhsRowOffsetsMemRefType = 
            MemRefType::get({lhsRows + 1}, rewriter.getIndexType());
        auto rhsMemRefType = mlir::MemRefType::get({rhsRows, rhsCols}, matrixElementType);
        auto resMemRefType = mlir::MemRefType::get({lhsRows, rhsCols}, matrixElementType);

        auto lhsValuesMemRef =
            rewriter.create<daphne::ConvertCSRMatrixToValuesMemRef>(loc, lhsValuesMemRefType, lhs);
        auto lhsColIdxsMemRef =
            rewriter.create<daphne::ConvertCSRMatrixToColIdxsMemRef>(loc, lhsColIdxsMemRefType, lhs);
        auto lhsRowOffsetsMemRef =
            rewriter.create<daphne::ConvertCSRMatrixToRowOffsetsMemRef>(loc, lhsRowOffsetsMemRefType, lhs);
        auto rhsMemRef = 
            rewriter.create<daphne::ConvertDenseMatrixToMemRef>(loc, rhsMemRefType, rhs);
        auto resMemRef = rewriter.create<memref::AllocOp>(loc, resMemRefType);

        auto zeroElement = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(matrixElementType));
        rewriter.create<linalg::FillOp>(loc, ValueRange{zeroElement}, ValueRange{resMemRef});

        auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
        auto numLhsRowsValue = rewriter.create<arith::ConstantIndexOp>(loc, lhsRows);
        auto numRhsColsValue = rewriter.create<arith::ConstantIndexOp>(loc, rhsCols);

        auto lhsRowLoop = rewriter.create<scf::ForOp>(
            loc, zero, numLhsRowsValue, one, ValueRange{},
            [&](OpBuilder &OpBuilderNested, Location locNested, Value loopIdx, ValueRange loopIterArgs) 
            {
                auto rowPtr = loopIdx;
                auto nextRowPtr = OpBuilderNested.create<arith::AddIOp>(locNested, rowPtr, one);
                auto rhsColLoop = OpBuilderNested.create<scf::ForOp>(
                    locNested, zero, numRhsColsValue, one, ValueRange{},
                    [&](OpBuilder &OpBuilderTwiceNested, Location locTwiceNested, Value loopIdx, ValueRange loopIterArgs) 
                    {
                        auto rhsCol = loopIdx;
                        auto lhsColIdxsLowerIncl = OpBuilderTwiceNested.create<memref::LoadOp>(
                            locTwiceNested, lhsRowOffsetsMemRef, ValueRange{rowPtr});
                        auto lhsColIdxsUpperExcl = OpBuilderTwiceNested.create<memref::LoadOp>(
                            locTwiceNested, lhsRowOffsetsMemRef, ValueRange{nextRowPtr});
                        
                        auto resValueLoop = OpBuilderTwiceNested.create<scf::ForOp>(
                            locTwiceNested, lhsColIdxsLowerIncl, lhsColIdxsUpperExcl, one, ValueRange{zeroElement},
                            [&](OpBuilder &OpBuilderThreetimesNested, Location locThreetimesNested, Value loopIdx, ValueRange loopIterArgs) 
                            {
                                auto lhsValue = OpBuilderThreetimesNested.create<memref::LoadOp>(
                                    locThreetimesNested, lhsValuesMemRef, ValueRange{loopIdx});
                                auto rhsRow = OpBuilderThreetimesNested.create<memref::LoadOp>(
                                    locThreetimesNested, lhsColIdxsMemRef, ValueRange{loopIdx});
                                auto rhsValue = OpBuilderThreetimesNested.create<memref::LoadOp>(
                                    locThreetimesNested, rhsMemRef, ValueRange{rhsRow, rhsCol});
                                
                                auto resValue = binaryWithConversionFunc<arith::MulIOp, arith::MulFOp>(
                                    OpBuilderThreetimesNested, locThreetimesNested, this->typeConverter, lhsValue, rhsValue);
                                
                                auto accResValue = binaryWithConversionFunc<arith::AddIOp, arith::AddFOp>(
                                    OpBuilderThreetimesNested, locThreetimesNested, this->typeConverter, loopIterArgs[0], resValue);
                                
                                OpBuilderThreetimesNested.create<scf::YieldOp>(locThreetimesNested, ValueRange{accResValue});
                            }
                        );
                        OpBuilderTwiceNested.create<memref::StoreOp>(
                            locTwiceNested, resValueLoop.getResult(0), resMemRef, ValueRange{rowPtr, rhsCol});    
                        OpBuilderTwiceNested.create<scf::YieldOp>(locTwiceNested, ValueRange{});
                    }
                );
                OpBuilderNested.create<scf::YieldOp>(locNested, ValueRange{});
            }
        );
        
        Value resDenseMatrix = convertMemRefToDenseMatrix(loc, rewriter, resMemRef, rhs.getType());
        rewriter.replaceOp(op, resDenseMatrix);
        return mlir::success();
    }
    
    LogicalResult matchAndRewrite(daphne::MatMulOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto loc = op->getLoc();
        mlir::daphne::MatrixType lhsMatrixType = adaptor.getLhs().getType().dyn_cast<mlir::daphne::MatrixType>();
        mlir::daphne::MatrixType rhsMatrixType = adaptor.getRhs().getType().dyn_cast<mlir::daphne::MatrixType>();

        if (lhsMatrixType.getRepresentation() == daphne::MatrixRepresentation::Sparse &&
            rhsMatrixType.getRepresentation() == daphne::MatrixRepresentation::Dense)
            return matchAndRewriteSparseDenseMat(op, adaptor, rewriter);

        if (lhsMatrixType.getRepresentation() == daphne::MatrixRepresentation::Sparse &&
            rhsMatrixType.getRepresentation() == daphne::MatrixRepresentation::Sparse)
            return matchAndRewriteSparseSparseMat(op, adaptor, rewriter);
        
        auto lhsRows = lhsMatrixType.getNumRows();
        auto lhsCols = lhsMatrixType.getNumCols();

        auto rhsRows = rhsMatrixType.getNumRows();
        auto rhsCols = rhsMatrixType.getNumCols();

        auto matrixElementType = lhsMatrixType.getElementType();

        // TODO(phil): if shape is unknown, e.g., row/col = -1 we currently
        // can't create a MemRefType
        auto lhsMemRefType = mlir::MemRefType::get({lhsRows, lhsCols}, matrixElementType);
        auto rhsMemRefType = mlir::MemRefType::get({rhsRows, rhsCols}, matrixElementType);

        mlir::MemRefType outputMemRefType = mlir::MemRefType::get({lhsRows, rhsCols}, matrixElementType);

        // daphne::Matrix -> memref
        mlir::Value lhs =
            rewriter.create<mlir::daphne::ConvertDenseMatrixToMemRef>(op->getLoc(), lhsMemRefType, adaptor.getLhs());
        mlir::Value rhs =
            rewriter.create<mlir::daphne::ConvertDenseMatrixToMemRef>(op->getLoc(), rhsMemRefType, adaptor.getRhs());

        // Alloc output memref
        mlir::Value outputMemRef = insertMemRefAlloc(outputMemRefType, loc, rewriter);

        // Fill the output MemRef
        if (matrixElementType.isIntOrIndex()) {
            auto signless_type = rewriter.getIntegerType(matrixElementType.getIntOrFloatBitWidth());
            auto fillValue =
                rewriter.create<arith::ConstantOp>(loc, signless_type, rewriter.getIntegerAttr(signless_type, 0));
            auto castedFillValue = this->typeConverter->materializeTargetConversion(rewriter, loc, matrixElementType,
                                                                                    mlir::ValueRange{fillValue});
            affineFillMemRef(castedFillValue, rewriter, loc, outputMemRefType.getShape(), op->getContext(),
                             outputMemRef);
        } else {
            mlir::Value fillValue = rewriter.create<mlir::arith::ConstantOp>(
                loc, matrixElementType, rewriter.getFloatAttr(matrixElementType, 0.0));
            affineFillMemRef(fillValue, rewriter, loc, outputMemRefType.getShape(), op->getContext(), outputMemRef);
        }
        // Do the actual MatMul with hand built codegen
        SmallVector<AffineForOp, 3> loops;
        if (options.vectorize && is_vectorizable(rhsMemRefType.getShape(), matrixElementType)) {
            vectorizedAffineMatMul(lhs, rhs, outputMemRef, rewriter, loc, lhsMemRefType.getShape(),
                                   rhsMemRefType.getShape(), op->getContext(), loops, matrixElementType,
                                   options.getVecSize(matrixElementType.getIntOrFloatBitWidth()));
        } else {
            affineMatMul(lhs, rhs, outputMemRef, rewriter, loc, lhsMemRefType.getShape(), rhsMemRefType.getShape(),
                         op->getContext(), loops, matrixElementType);
        }
        if (options.tile && is_tileable(rhsMemRefType.getShape())) {
            auto tile_sizes = extendTileSizes(lhsRows);
            if (!options.useFixedTileSizes) {
                tile_sizes = getTileSizesFromCache(matrixElementType, loops[1].getStep(), lhsRows);
            }
            tile_loops(loc, loops, tile_sizes);
        } else if (options.invert_loops) {
            permuteLoops(loops, {0, 2, 1});
        }
        mlir::Value DM = convertMemRefToDenseMatrix(loc, rewriter, outputMemRef, op.getType());

        rewriter.replaceOp(op, DM);
        return success();
    }

    // tile_loops requires 5 tile sizes. If fewer tile sizes are specified, we
    // can extend with the size of the loop, since loops with only one iteration
    // are later removed.
    SmallVector<unsigned, 5> extendTileSizes(int64_t max_loop_length) const {
        SmallVector<unsigned, 5> tile_sizes = options.tile_sizes;
        while (tile_sizes.size() < 5) {
            tile_sizes.push_back(max_loop_length);
        }
        return tile_sizes;
    }

    // Choose tile sizes so that reuse is happening across the cache levels.
    // This is just a proof of concept and not a very sophisticated strategy.
    // Assuming cache sizes are in Bytes not KB or other units. Assume square
    // matmul of length loop_length. The target below is laid out assuming there
    // are a number of vector registers available. If not all cache sizes "move
    // down" a slot if set. If there are also no cache sizes available, set MR
    // and NR to 2, since otherwise the tiling breaks. Target:  MR * NR ~
    // Register size * 3 / 4
    //          KC * NR ~ L1,
    //          MC * KC ~ L2,
    //          NC * MC ~ L3
    //          & NR divides NC & MR divides MC
    SmallVector<unsigned, 5> getTileSizesFromCache(Type const matrixElementType, int64_t vec_size,
                                                   int64_t loop_length) const {
        SmallVector<unsigned, 5> tile_sizes;
        int bitwidth = matrixElementType.getIntOrFloatBitWidth();
        int register_size = options.getRegisterSize();
        int no_register = 0;
        if (register_size == 1) {
            if (options.cache_sizes.size() > 0) {
                tile_sizes.push_back(std::max(2, (int)(std::sqrt(register_size / bitwidth))));
                tile_sizes.push_back(tile_sizes.back());
                no_register++;
            } else {
                tile_sizes.push_back(2);
                tile_sizes.push_back(2);
            }
        } else {
            tile_sizes.push_back(std::max(2, (int)(std::sqrt(register_size / bitwidth * 3 / 4))));
            tile_sizes.push_back(tile_sizes.back());
        }
        if (options.cache_sizes.size() > 0) {
            int idx = 0;
            for (auto cache_size = options.cache_sizes.begin() + no_register; cache_size != options.cache_sizes.end();
                 cache_size++) {
                unsigned candidate = std::max(1, (int)(*cache_size / tile_sizes.back() / bitwidth));
                if (idx == 3)
                    candidate = candidate - (candidate % tile_sizes[0]);
                if (idx == 4)
                    candidate = candidate - (candidate % tile_sizes[1]);
                tile_sizes.push_back(candidate);
                idx++;
            }
        }
        while (tile_sizes.size() < 5) {
            tile_sizes.push_back(loop_length);
        }
        // If vector size is longer than 1, we need to keep that in mind for the
        // NR loop
        if (vec_size > 1)
            tile_sizes[1] = std::max(1, (int)(tile_sizes[1] / vec_size));
        return tile_sizes;
    }

    // Tile the affine loop nest generated from MatMulOp with the specified tile
    // sizes. Includes validations to follow the movement and creation of the
    // tile loops.
    void tile_loops(mlir::Location loc, SmallVector<AffineForOp, 3> loops, SmallVector<unsigned, 5> tile_sizes) const {
        unsigned NC = tile_sizes[4];
        unsigned MC = tile_sizes[3];
        unsigned KC = tile_sizes[2];
        unsigned NR = tile_sizes[1];
        unsigned MR = tile_sizes[0];
        unsigned KU = options.unroll_factor;
        [[maybe_unused]] auto vec_size = loops[1].getStep();
        llvm::SmallVector<AffineForOp> loopNest;
        getPerfectlyNestedLoops(loopNest, loops.front());
        // tile i with MC, j with NC, k with KC
        llvm::SmallVector<AffineForOp> tiledNest;
        if (failed(tilePerfectlyNested(loopNest, {MC, NC, KC}, &tiledNest))) {
            spdlog::warn("Could not tile the loop nest in MatMulLowering");
        };

#define GEN_ERR_MSG(name, size, expected)                                                                              \
    std::string(name) + " should have step size " + std::string(expected) + " but is " + std::to_string(size)

        if (tiledNest[0].getStep() != MC)
            throw ErrorHandler::compilerError(
                loc, "MatMulOpLowering (tile_loops)",
                GEN_ERR_MSG("tiledNest 0", tiledNest[0].getStep(), "MC (" + std::to_string(MC) + ")"));
        if (tiledNest[1].getStep() != NC * vec_size)
            throw ErrorHandler::compilerError(loc, "MatMulOpLowering (tile_loops)",
                                              GEN_ERR_MSG("tiledNest 1", tiledNest[1].getStep(),
                                                          "NC * vec_size (" + std::to_string(NC * vec_size) + ")"));
        if (tiledNest[2].getStep() != KC)
            throw ErrorHandler::compilerError(
                loc, "MatMulOpLowering (tile_loops)",
                GEN_ERR_MSG("tiledNest 2", tiledNest[2].getStep(), "KC (" + std::to_string(KC) + ")"));
        if (tiledNest[3].getStep() != 1)
            throw ErrorHandler::compilerError(loc, "MatMulOpLowering (tile_loops)",
                                              GEN_ERR_MSG("tiledNest 3", tiledNest[3].getStep(), "1"));
        if (tiledNest[4].getStep() != vec_size)
            throw ErrorHandler::compilerError(
                loc, "MatMulOpLowering (tile_loops)",
                GEN_ERR_MSG("tiledNest 4", tiledNest[4].getStep(), "vec_size (" + std::to_string(vec_size) + ")"));
        if (tiledNest[5].getStep() != 1)
            throw ErrorHandler::compilerError(loc, "MatMulOpLowering (tile_loops)",
                                              GEN_ERR_MSG("tiledNest 5", tiledNest[5].getStep(), "1"));

        // Further tile the i mod MC loop with MR
        if (failed(tilePerfectlyNested(tiledNest[3], {MR}))) {
            spdlog::warn("Could not tile the second i loop in MatMulLowering");
        };

        // Further tile the j mod NC loop with NR
        if (tiledNest[4].getStep() != vec_size)
            throw ErrorHandler::compilerError(
                loc, "MatMulOpLowering (tile_loops)",
                GEN_ERR_MSG("tiledNest 4", tiledNest[4].getStep(), "vec_size (" + std::to_string(vec_size) + ")"));
        if (failed(tilePerfectlyNested(tiledNest[4], {NR}))) {
            spdlog::warn("Could not tile the second j loop in MatMulLowering");
        };

        llvm::SmallVector<AffineForOp> twiceTiledNest;
        getPerfectlyNestedLoops(twiceTiledNest, tiledNest[0]);
        // i loops
        if (twiceTiledNest[0].getStep() != MC)
            throw ErrorHandler::compilerError(
                loc, "MatMulOpLowering (tile_loops)",
                GEN_ERR_MSG("twiceTiledNest 0", twiceTiledNest[0].getStep(), "MC (" + std::to_string(MC) + ")"));
        if (twiceTiledNest[3].getStep() != MR)
            throw ErrorHandler::compilerError(
                loc, "MatMulOpLowering (tile_loops)",
                GEN_ERR_MSG("twiceTiledNest 3", twiceTiledNest[3].getStep(), "MR (" + std::to_string(MR) + ")"));
        if (twiceTiledNest[4].getStep() != 1)
            throw ErrorHandler::compilerError(loc, "MatMulOpLowering (tile_loops)",
                                              GEN_ERR_MSG("twiceTiledNest 4", twiceTiledNest[4].getStep(), "1"));

        // j loops
        if (twiceTiledNest[1].getStep() != NC * vec_size)
            throw ErrorHandler::compilerError(loc, "MatMulOpLowering (tile_loops)",
                                              GEN_ERR_MSG("twiceTiledNest 1", twiceTiledNest[1].getStep(),
                                                          "NC * vec_size (" + std::to_string(NC * vec_size) + ")"));
        if (twiceTiledNest[5].getStep() != NR * vec_size)
            throw ErrorHandler::compilerError(loc, "MatMulOpLowering (tile_loops)",
                                              GEN_ERR_MSG("twiceTiledNest 5", twiceTiledNest[5].getStep(),
                                                          "NR * vec_size (" + std::to_string(NR * vec_size) + ")"));
        if (twiceTiledNest[6].getStep() != vec_size)
            throw ErrorHandler::compilerError(loc, "MatMulOpLowering (tile_loops)",
                                              GEN_ERR_MSG("twiceTiledNest 6", twiceTiledNest[6].getStep(),
                                                          "vec_size (" + std::to_string(vec_size) + ")"));

        // k loops
        if (twiceTiledNest[2].getStep() != KC)
            throw ErrorHandler::compilerError(
                loc, "MatMulOpLowering (tile_loops)",
                GEN_ERR_MSG("twiceTiledNest 2", twiceTiledNest[2].getStep(), "KC (" + std::to_string(KC) + ")"));
        if (twiceTiledNest[7].getStep() != 1)
            throw ErrorHandler::compilerError(loc, "MatMulOpLowering (tile_loops)",
                                              GEN_ERR_MSG("twiceTiledNest 7", twiceTiledNest[7].getStep(), "1"));

        // permute loops to final order (i / MC, j / NC, k / KC, i / MR, i mod
        // MR, j / NR, j mod NR, k mod KC) ->
        //                              (j / NC, k / KC, i / MC, j / NR, i / MR,
        //                              k mod KC, j mod NR, i mod MR)
        unsigned root_idx = permuteLoops(twiceTiledNest, {2, 0, 1, 4, 7, 3, 6, 5});

        // Unroll and jam
        llvm::SmallVector<AffineForOp> blisTiledLoops;
        getPerfectlyNestedLoops(blisTiledLoops, twiceTiledNest[root_idx]);
        // i loops
        if (blisTiledLoops[2].getStep() != MC)
            throw ErrorHandler::compilerError(
                loc, "MatMulOpLowering (tile_loops)",
                GEN_ERR_MSG("blisTiled 2", blisTiledLoops[2].getStep(), "MC (" + std::to_string(MC) + ")"));
        if (blisTiledLoops[4].getStep() != MR)
            throw ErrorHandler::compilerError(
                loc, "MatMulOpLowering (tile_loops)",
                GEN_ERR_MSG("blisTiled 4", blisTiledLoops[4].getStep(), "MR (" + std::to_string(MR) + ")"));
        if (blisTiledLoops[7].getStep() != 1)
            throw ErrorHandler::compilerError(loc, "MatMulOpLowering (tile_loops)",
                                              GEN_ERR_MSG("blisTiled 7", blisTiledLoops[7].getStep(), "1"));

        // j loops
        if (blisTiledLoops[0].getStep() != NC * vec_size)
            throw ErrorHandler::compilerError(loc, "MatMulOpLowering (tile_loops)",
                                              GEN_ERR_MSG("blisTiled 0", blisTiledLoops[0].getStep(),
                                                          "NC * vec_size (" + std::to_string(NC * vec_size) + ")"));
        if (blisTiledLoops[3].getStep() != NR * vec_size)
            throw ErrorHandler::compilerError(loc, "MatMulOpLowering (tile_loops)",
                                              GEN_ERR_MSG("blisTiled 3", blisTiledLoops[3].getStep(),
                                                          "NR * vec_size (" + std::to_string(NR * vec_size) + ")"));
        if (blisTiledLoops[6].getStep() != vec_size)
            throw ErrorHandler::compilerError(
                loc, "MatMulOpLowering (tile_loops)",
                GEN_ERR_MSG("blisTiled 6", blisTiledLoops[6].getStep(), "vec_size (" + std::to_string(vec_size) + ")"));

        // k loops
        if (blisTiledLoops[1].getStep() != KC)
            throw ErrorHandler::compilerError(
                loc, "MatMulOpLowering (tile_loops)",
                GEN_ERR_MSG("blisTiled 1", blisTiledLoops[1].getStep(), "KC (" + std::to_string(KC) + ")"));
        if (blisTiledLoops[5].getStep() != 1)
            throw ErrorHandler::compilerError(loc, "MatMulOpLowering (tile_loops)",
                                              GEN_ERR_MSG("blisTiled 5", blisTiledLoops[5].getStep(), "1"));

#undef GEN_ERR_MSG

        // Unroll jam causes Segfault, if called in a way where the loop is not
        // cleanly divided.
        if (options.unroll_jam_factor > 0 && blisTiledLoops[5].getUpperBound().getMap().getNumResults() == 1 &&
            succeeded(loopUnrollJamUpToFactor(blisTiledLoops[5], options.unroll_jam_factor))) {
            if (blisTiledLoops[6].getUpperBound().getMap().getNumResults() != 1 ||
                failed(loopUnrollJamUpToFactor(blisTiledLoops[6], options.unroll_jam_factor))) {
                spdlog::warn("Could not unroll the (j mod NC) mod NR loop in "
                             "MatMulLowering");
            }
        } else {
            spdlog::warn("Could not unroll the (i mod MC) mod MR loop in "
                         "MatMulLowering");
        }

        llvm::SmallVector<AffineForOp> lastNest;
        getPerfectlyNestedLoops(lastNest, blisTiledLoops.front());
        int64_t i = 0;
        while (succeeded(promoteIfSingleIteration(lastNest[i])) && i < 4) {
            i++;
        }

        if (KU > 0 && failed(loopUnrollUpToFactor(lastNest.back(), KU))) {
            spdlog::warn("Could not unroll the K loop in MatMulLowering");
        }
    }
};

namespace {
/**
 * @brief The MatMulLoweringPass rewrites the MatMulOp from the DaphneDialect
 * to a affine loop structure implementing a multi tiled loop structure.
 * Lowering can be performed with or without
 *  - vectorization
 *  - tiling
 * The vector size is specifies in bits and then adapted to the value type in
 * the Operation, but stores at least one element. The tile sizes can be fixed
 * or attempted to be generated automatically. The pass options are specified
 * and have descriptions in Passes.td.
 *
 * A more detailed description can be found in 'daphneir/Passes.td'.
 */
struct MatMulLoweringPass : public impl::MatMulOpLoweringPassBase<MatMulLoweringPass> {
    MatMulLoweringPass() = default;

  public:
    explicit MatMulLoweringPass(bool matmul_tile, int matmul_vec_size_bits,
                                std::vector<unsigned> matmul_fixed_tile_sizes, bool matmul_use_fixed_tile_sizes,
                                int matmul_unroll_factor, int matmul_unroll_jam_factor, int matmul_num_vec_registers,
                                bool matmul_invert_loops)
        : impl::MatMulOpLoweringPassBase<MatMulLoweringPass>() {
        this->matmul_tile = matmul_tile;
        this->matmul_vec_size_bits = matmul_vec_size_bits;
        this->matmul_fixed_tile_sizes = matmul_fixed_tile_sizes;
        this->matmul_use_fixed_tile_sizes = matmul_use_fixed_tile_sizes;
        this->matmul_unroll_factor = matmul_unroll_factor;
        this->matmul_unroll_jam_factor = matmul_unroll_jam_factor;
        this->matmul_num_vec_registers = matmul_num_vec_registers;
        this->matmul_invert_loops = matmul_invert_loops;
    }

    void runOnOperation() override;

  private:
    // Get the L1, L2 and L3 cache sizes to adapt tile sizes.
    // So far assumes process is executed on a single processing unit.
    // See example:
    // https://www.open-mpi.org/projects/hwloc/doc/v2.2.0/a00324.php#cli_examples
    SmallVector<int> get_cache_sizes() const {
        hwloc_topology_t topology;
        hwloc_obj_t obj;
        SmallVector<int> sizes;

        // Allocate and initialize topology object
        hwloc_topology_init(&topology);
        // Perform topology detection
        hwloc_topology_load(topology);

        for (obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_PU, 0); obj; obj = obj->parent)
            if (hwloc_obj_type_is_cache(obj->type)) {
                sizes.push_back(obj->attr->cache.size);
            }
        return sizes;
    }
};
} // end anonymous namespace

void MatMulLoweringPass::runOnOperation() {
    auto module = getOperation();
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
    target.addLegalDialect<mlir::vector::VectorDialect>();
    target.addLegalDialect<daphne::DaphneDialect>();
    target.addLegalDialect<BuiltinDialect>();
    target.addLegalDialect<math::MathDialect>();

    target.addLegalOp<mlir::daphne::ConvertDenseMatrixToMemRef>();
    target.addLegalOp<mlir::daphne::ConvertMemRefToDenseMatrix>();
    target.addLegalOp<mlir::daphne::DecRefOp>();
    LowerMatMulOpOptions options;
    if (matmul_tile) {
        options.enableTiling();
        if (matmul_use_fixed_tile_sizes) {
            options.useFixedTileSizes = true;
            options.setTileSizes(matmul_fixed_tile_sizes);
        } else {
            options.setCacheSizes(get_cache_sizes());
        }
        options.setUnrollFactor(matmul_unroll_factor);
        options.setUnrollJamFactor(matmul_unroll_jam_factor);
    }
    if (matmul_vec_size_bits > 0) {
        options.enableVectorization();
        options.setVectorSizeBits(matmul_vec_size_bits);
    }
    options.enableLoopInversion(matmul_invert_loops);
    options.setNumberOfVectorRegisters(matmul_num_vec_registers);
    target.addDynamicallyLegalOp<mlir::daphne::MatMulOp>(
        [options](Operation *op) { return !is_valid_options(options); });

    patterns.insert<MatMulLowering>(typeConverter, &getContext(), options);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::daphne::createMatMulOpLoweringPass(
    bool matmul_tile, int matmul_vec_size_bits, std::vector<unsigned> matmul_fixed_tile_sizes,
    bool matmul_use_fixed_tile_sizes, int matmul_unroll_factor, int matmul_unroll_jam_factor,
    int matmul_num_vec_registers, bool matmul_invert_loops) {
    return std::make_unique<MatMulLoweringPass>(
        matmul_tile, matmul_vec_size_bits, matmul_fixed_tile_sizes, matmul_use_fixed_tile_sizes, matmul_unroll_factor,
        matmul_unroll_jam_factor, matmul_num_vec_registers, matmul_invert_loops);
}

// This is used by daphne-opt and automatically inserts the options provided on
// the command line into the pass.
std::unique_ptr<OperationPass<ModuleOp>> mlir::daphne::createMatMulOpLoweringPass() {
    return std::make_unique<MatMulLoweringPass>();
}
