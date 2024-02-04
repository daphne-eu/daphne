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
#include <cassert>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "api/cli/DaphneUserConfig.h"
#include "compiler/utils/LoweringUtils.h"
#include "hwloc.h"
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DECL_MATMULOPLOWERINGPASS
#define GEN_PASS_DEF_MATMULOPLOWERINGPASS
#include "ir/daphneir/Passes.h.inc"
} // namespace mlir

using namespace mlir;

static constexpr int ROW = 0;
static constexpr int COL = 1;

struct LowerMatMulOpOptions {
    LowerMatMulOpOptions () {}
    int vec_size_bits{0};
    bool vectorize{false};
    bool tile{false};
    bool useFixedTileSizes{false};
    int register_size{4*4*64};
    llvm::SmallVector<int, 3> cache_sizes;
    llvm::SmallVector<unsigned, 5> tile_sizes;
    int unroll_factor{1};
    // 4 is the default value for the affine unroll jam
    int unroll_jam_factor{4};

    LowerMatMulOpOptions &setTileSizes(std::vector<unsigned> sizes) {
        tile_sizes.clear();
        for (auto s : sizes) {
            tile_sizes.push_back(s);
        }  
        return *this;
    }
    LowerMatMulOpOptions &setUnrollFactor(int f) {
        if (f < 1) throw std::invalid_argument("Unroll factor must be an integer larger than 0.");
        unroll_factor = f;
        return *this;
    }
    LowerMatMulOpOptions &setUnrollJamFactor(int f) {
        if (f < 1) throw std::invalid_argument("Unroll jam factor must be an integer larger than 0.");
        unroll_jam_factor = f;
        return *this;
    }
    LowerMatMulOpOptions &setCacheSizes(llvm::SmallVector<int> caches) {
        cache_sizes.clear();
        for (auto c:caches) {
            cache_sizes.push_back(c);
        }
        return *this;
    }
    LowerMatMulOpOptions &enableVectorization(bool b = true) {
        vectorize = b;
        return *this;
    }
    LowerMatMulOpOptions &setVectorSizeBits(int s) {
        if (s < 0) throw std::invalid_argument("Vector size bits must be larger or equal to 0.");
        vec_size_bits = s;
        return *this;
    }
    LowerMatMulOpOptions &enableTiling(bool b = true) {
        tile = b;
        return *this;
    }
    int getVecSize(int bitwidth) const {
        if (vec_size_bits > 0) {
            return std::max(1, vec_size_bits / bitwidth);
        }
        else {
            return 1;
        }
    }
};

llvm::SmallVector<AffineForOp, 3> affineMatMul(mlir::Value &lhs, mlir::Value &rhs, mlir::Value &output,
                  ConversionPatternRewriter &rewriter, mlir::Location loc,
                  ArrayRef<int64_t> lhsShape, ArrayRef<int64_t> rhsShape,
                  mlir::MLIRContext *ctx, SmallVector<AffineForOp, 3> &loops) {    
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

    
        auto  a = rewriter.create<AffineLoadOp>(loc, lhs,
                    ValueRange{rowLoop.getInductionVar(), fmaLoop.getInductionVar()});
        auto  b = rewriter.create<AffineLoadOp>(
                   loc, rhs,
                    ValueRange{fmaLoop.getInductionVar(), colLoop.getInductionVar()});
        auto  c = rewriter.create<AffineLoadOp>(
                    loc, output,
                    ValueRange{rowLoop.getInductionVar(), colLoop.getInductionVar()});
        // split out into add and multiply
        Value res = rewriter.create<LLVM::FMAOp>(loc, a, b, c);
       
        rewriter.create<AffineStoreOp>(loc, res, output,
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


llvm::SmallVector<AffineForOp, 3> vectorizedAffineMatMul(mlir::Value &lhs, mlir::Value &rhs, mlir::Value &output,
                  ConversionPatternRewriter &rewriter, mlir::Location loc,
                  ArrayRef<int64_t> lhsShape, ArrayRef<int64_t> rhsShape,
                  mlir::MLIRContext *ctx, llvm::SmallVector<AffineForOp, 3> &loops, Type elementType, int64_t vec_size) {
    auto vec_Type = mlir::VectorType::get({vec_size}, elementType);
        
    // TODO: We need an option to enable smaller vector sizes for the ends of each row.
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

class MatMulLowering : public OpConversionPattern<daphne::MatMulOp> {
    const LowerMatMulOpOptions options;
    std::shared_ptr<spdlog::logger> logger;
   public:
    using OpConversionPattern::OpConversionPattern;
    explicit MatMulLowering(MLIRContext *context, LowerMatMulOpOptions const &options) 
        : OpConversionPattern(context, PatternBenefit(1)), options(options) {
            logger = spdlog::get("compiler");
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

    // TODO: Test why not
    // Cannot tile non square matmul.
    bool is_tileable(ArrayRef<int64_t> const rhsShape) const {
        if (rhsShape[COL] != rhsShape[ROW]) {
            return false;
        }
        return true;
    }
    
    LogicalResult matchAndRewrite(
        daphne::MatMulOp op, OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override {
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
        // Cannot lower integer matmul, because we cannot create Memrefs with integer type at the moment.
        if (matrixElementType.isIntOrIndex()) return failure();

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
        SmallVector<AffineForOp, 3> loops;
        if (options.vectorize && is_vectorizable(rhsMemRefType.getShape(), matrixElementType)) {
            vectorizedAffineMatMul(lhs, rhs, outputMemRef, rewriter, loc,
                     lhsMemRefType.getShape(), rhsMemRefType.getShape(),
                     op->getContext(), loops, matrixElementType, 
                     options.getVecSize(matrixElementType.getIntOrFloatBitWidth()));
        } else {
            affineMatMul(lhs, rhs, outputMemRef, rewriter, loc,
                     lhsMemRefType.getShape(), rhsMemRefType.getShape(),
                     op->getContext(), loops);
        }
        if (options.tile && is_tileable(rhsMemRefType.getShape())) {
                auto tile_sizes = extendTileSizes(lhsRows);
            if (!options.useFixedTileSizes) {
                tile_sizes = getTileSizesFromCache(matrixElementType, loops[1].getStep(), lhsRows);
            }
            tile_loops(loops, tile_sizes);
        }

        mlir::Value DM = convertMemRefToDenseMatrix(loc, rewriter, outputMemRef,
                                                    op.getType());

        rewriter.replaceOp(op, DM);
        return success();
    }

    // tile_loops requires 5 tile sizes. If fewer tile sizes are specified, we can extend with the size of
    // the loop, since loops with only one iteration are later removed.
    SmallVector<unsigned, 5> extendTileSizes(int64_t max_loop_length) const {
        SmallVector<unsigned, 5> tile_sizes = options.tile_sizes;
        while (tile_sizes.size() < 5) {
            tile_sizes.push_back(max_loop_length);
        }
        return tile_sizes;
    }

    // Choose tile sizes so that reuse is happening across the cache levels. This is just a proof of concept and not a very
    // sophisticated strategy.
    // Assuming cache sizes are in Bytes not KB or other units.
    // Register size is currently hard coded in the MatMulLoweringOptions.
    // Target:  MR * NR ~ Register size
    //          KC * NR ~ L1,
    //          MC * KC ~ L2,
    //          NC * MC ~ L3 
    SmallVector<unsigned, 5> getTileSizesFromCache(Type const matrixElementType, int64_t vec_size, int64_t loop_length) const {
        SmallVector<unsigned, 5> tile_sizes;
        int bitwidth = matrixElementType.getIntOrFloatBitWidth();
        tile_sizes.push_back(std::max(1, (int)(std::sqrt(options.register_size / bitwidth))));
        tile_sizes.push_back(tile_sizes.back());
        if (options.cache_sizes.size() > 0) {
            for (auto cache_size=options.cache_sizes.begin(); cache_size != options.cache_sizes.end(); cache_size++) {
                tile_sizes.push_back(std::max(1, (int)(*cache_size / tile_sizes.back() / bitwidth)));
            }
        }
        while (tile_sizes.size() < 5) {
            tile_sizes.push_back(loop_length);
        }
        // If vector size is longer than 1, we need to keep that in mind for the NR loop
        if (vec_size > 1) tile_sizes[1] = std::max(1, (int)(tile_sizes[1] / vec_size));
        return tile_sizes;
    }

    // Tile the affine loop nest generated from MatMulOp with the specified tile sizes.
    // Includes asserts to follow the movement and creation of the tile loops.
    void tile_loops(SmallVector<AffineForOp, 3> loops, SmallVector<unsigned, 5> tile_sizes) const {
        unsigned NC = tile_sizes[4];
        unsigned MC = tile_sizes[3];
        unsigned KC = tile_sizes[2];
        unsigned NR = tile_sizes[1];
        unsigned MR = tile_sizes[0];
        unsigned KU = options.unroll_factor;
        auto vec_size = loops[1].getStep();
        llvm::SmallVector<AffineForOp> loopNest;
        getPerfectlyNestedLoops(loopNest, loops.front());
        // tile i with MC, j with NC, k with KC
        llvm::SmallVector<AffineForOp> tiledNest;
        if (failed(tilePerfectlyNested(loopNest, {MC, NC, KC}, &tiledNest))) {
            if(logger->should_log(spdlog::level::debug)) {
                std::string s;
                llvm::raw_string_ostream stream(s);
                logger->debug("Could not tile the loop nest in MatMulLowering", s);
            }
        };
        assert(tiledNest[0].getStep() == MC && "0 should have step size MC.");
        assert(tiledNest[1].getStep() == NC * vec_size && "1 should have step size NC * vec_size.");
        assert(tiledNest[2].getStep() == KC && "2 should have step size KC.");
        assert(tiledNest[3].getStep() == 1 && "3 should have step size 1.");
        assert(tiledNest[4].getStep() == 1 * vec_size && "4 should have step size vec_size.");
        assert(tiledNest[5].getStep() == 1 && "5 should have step size 1.");

        // Further tile the i mod MC loop with MR
        if (failed(tilePerfectlyNested(tiledNest[3], {MR}))) {
            if(logger->should_log(spdlog::level::debug)) {
                std::string s;
                llvm::raw_string_ostream stream(s);
                logger->debug("Could not tile the second i loop in MatMulLowering", s);
            }
        };
        
        // Further tile the j mod NC loop with NR
        assert(tiledNest[4].getStep() == 1 * vec_size && "4 should have step size vec_size.");
        if (failed(tilePerfectlyNested(tiledNest[4], {NR}))) {
            if(logger->should_log(spdlog::level::debug)) {
                std::string s;
                llvm::raw_string_ostream stream(s);
                logger->debug("Could not tile the second j loop in MatMulLowering", s);
            }
        };


        llvm::SmallVector<AffineForOp> twiceTiledNest;
        getPerfectlyNestedLoops(twiceTiledNest, tiledNest[0]);
        assert(twiceTiledNest[0].getStep() == MC && "tTN: 0 should have step size MC.");  // i loops
        assert(twiceTiledNest[3].getStep() == MR && "tTN: 3 should have step size MR.");
        assert(twiceTiledNest[4].getStep() == 1 && "tTN: 4 should have step size 1.");
        assert(twiceTiledNest[1].getStep() == NC * vec_size && "tTN: 1 should have step size NC * vec_size.");  // j loops
        assert(twiceTiledNest[5].getStep() == NR * vec_size && "tTN: 5 should have step size NR.");
        assert(twiceTiledNest[6].getStep() == 1 * vec_size && "tTN: 6 should have step size vec_size.");
        assert(twiceTiledNest[2].getStep() == KC && "tTN: 2 should have step size 1.");  // k loops
        assert(twiceTiledNest[7].getStep() == 1 && "tTN: 7 should have step size 1."); 
                               
        // permute loops to final order (i / MC, j / NC, k / KC, i / MR, i mod MR, j / NR, j mod NR, k mod KC) ->
        //                              (j / NC, k / KC, i / MC, j / NR, i / MR, k mod KC, j mod NR, i mod MR)
        // TODO: This assert only fails in debug mode?!
        //assert(isValidLoopInterchangePermutation(twiceTiledNest, {2, 0, 1, 4, 7, 3, 6, 5}));
        unsigned root_idx = permuteLoops(twiceTiledNest, {2, 0, 1, 4, 7, 3, 6, 5});

        // Unroll and jam
        llvm::SmallVector<AffineForOp> blisTiledLoops;
        getPerfectlyNestedLoops(blisTiledLoops, twiceTiledNest[root_idx]); 
        assert(blisTiledLoops[2].getStep() == MC && "blisTiled: 2 should have step size MC.");  // i loops
        assert(blisTiledLoops[4].getStep() == MR && "blisTiled: 4 should have step size MR.");
        assert(blisTiledLoops[7].getStep() == 1 && "blisTiled: 7 should have step size 1.");
        assert(blisTiledLoops[0].getStep() == NC * vec_size && "blisTiled: 0 should have step size NC * vec_size.");  // j loops
        assert(blisTiledLoops[3].getStep() == NR * vec_size && "blisTiled: 3 should have step size NR * vec_size.");
        assert(blisTiledLoops[6].getStep() == vec_size && "blisTiled: 6 should have step size vec_size.");
        assert(blisTiledLoops[1].getStep() == KC && "blisTiled: 1 should have step size 1.");  // k loops
        assert(blisTiledLoops[5].getStep() == 1 && "blisTiled: 5 should have step size 1.");
        // Unroll jam causes Segfault, if called in a way where the loop is not cleanly divided.
        if (blisTiledLoops[5].getUpperBound().getMap().getNumResults() == 1) {
            (void)loopUnrollJamUpToFactor(blisTiledLoops[5], options.unroll_jam_factor);
            Block &block = blisTiledLoops[5].getRegion().front();
            auto itForOp = block.getOps<AffineForOp>();
            for (auto f : itForOp) {
                if (f.getUpperBound().getMap().getNumResults() == 1) {
                    (void)loopUnrollJamUpToFactor(f,options.unroll_jam_factor);
                }
            }
        }
        
        llvm::SmallVector<AffineForOp> lastNest;
        getPerfectlyNestedLoops(lastNest, blisTiledLoops.front()); 
        
        if (failed(loopUnrollUpToFactor(lastNest.back(), KU))) {
            if(logger->should_log(spdlog::level::debug)) {
                std::string s;
                llvm::raw_string_ostream stream(s);
                logger->debug("Could not unroll the K loop in MatMulLowering", s);
            }
        }
        int64_t i = 0;
        while (succeeded(promoteIfSingleIteration(lastNest[i])) && i < 4) {
            i++;
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
 * The vector size is specifies in bits and then adapted to the value type in the Operation, 
 * but stores at least one element. The tile sizes can be fixed or attempted to be generated automatically.
 * The pass options are specified and have descriptions in Passes.td. 
 *
 * A more detailed description can be found in 'daphneir/Passes.td'.
 */
struct MatMulLoweringPass
        : public impl::MatMulOpLoweringPassBase<MatMulLoweringPass> {
    MatMulLoweringPass() = default;
    public:
        explicit MatMulLoweringPass(bool matmul_tile, int matmul_vec_size_bits, std::vector<unsigned> matmul_fixed_tile_sizes,
        bool matmul_use_fixed_tile_sizes, int matmul_unroll_factor, int matmul_unroll_jam_factor) : impl::MatMulOpLoweringPassBase<MatMulLoweringPass>() {
            this->matmul_tile = matmul_tile;
            this->matmul_vec_size_bits = matmul_vec_size_bits;
            this->matmul_fixed_tile_sizes = matmul_fixed_tile_sizes;
            this->matmul_use_fixed_tile_sizes = matmul_use_fixed_tile_sizes;
            this->matmul_unroll_factor = matmul_unroll_factor;
            this->matmul_unroll_jam_factor = matmul_unroll_jam_factor;
        }
        
        void runOnOperation() override;

    private:
        // Get the L1, L2 and L3 cache sizes to adapt tile sizes.
        // So far assumes process is executed on a single processing unit.
        // See example: https://www.open-mpi.org/projects/hwloc/doc/v2.2.0/a00324.php#cli_examples
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
        target.addDynamicallyLegalOp<mlir::daphne::MatMulOp>(
        [](Operation *op) {
            return op->getOperandTypes()[0].dyn_cast<mlir::daphne::MatrixType>().getElementType().isIntOrIndex();
        });

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
        
        patterns.insert<MatMulLowering>(&getContext(), options);
        
        if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::daphne::createMatMulOpLoweringPass(bool matmul_tile,
        int matmul_vec_size_bits,
        std::vector<unsigned> matmul_fixed_tile_sizes,
        bool matmul_use_fixed_tile_sizes,
        int matmul_unroll_factor,
        int matmul_unroll_jam_factor) {
    return std::make_unique<MatMulLoweringPass>(matmul_tile,
        matmul_vec_size_bits,
        matmul_fixed_tile_sizes,
        matmul_use_fixed_tile_sizes,
        matmul_unroll_factor,
        matmul_unroll_jam_factor);
}

// This is used by daphne-opt and automatically inserts the options provided on the command line into the pass.
std::unique_ptr<OperationPass<ModuleOp>>  mlir::daphne::createMatMulOpLoweringPass() {
    return std::make_unique<MatMulLoweringPass>();
}
