#include "compiler/conversion/DaphneToLinalg.h"
#include "compiler/conversion/AggReductions.h"
#include "ir/daphneir/Daphne.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace mlir;

// TODO: Will need to be templated to work with all elementwise binary ops.
struct EwMulOpConverter : public OpConversionPattern<daphne::EwMulOp> {
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(daphne::EwMulOp op, OpAdaptor adaptor, ConversionPatternRewriter &rw) const override {
        auto resTy = dyn_cast<RankedTensorType>(typeConverter->convertType(op.getType()));
        auto lhsTy = dyn_cast<RankedTensorType>(adaptor.getLhs().getType());
        if (!resTy || !lhsTy)
            return rw.notifyMatchFailure(op, "could not convert input or result type");
        Location loc = op.getLoc();
        SmallVector<Value> dynDims;
        for (unsigned i = 0; i < resTy.getRank(); ++i)
            if (resTy.isDynamicDim(i))
                dynDims.push_back(rw.create<tensor::DimOp>(loc, adaptor.getLhs(), i));
        Value init = rw.create<tensor::EmptyOp>(loc, resTy, dynDims);
        auto idMap = rw.getMultiDimIdentityMap(resTy.getRank());
        auto [rhs, rhsMap] = prepareRhs(op, lhsTy, resTy, idMap, adaptor.getRhs(), rw, loc);
        SmallVector<utils::IteratorType> iters(resTy.getRank(), utils::IteratorType::parallel);
        auto generic = rw.create<linalg::GenericOp>(
            loc, resTy, ValueRange{adaptor.getLhs(), rhs}, ValueRange{init}, ArrayRef<AffineMap>{idMap, rhsMap, idMap},
            iters, [&](OpBuilder &b, Location nl, ValueRange args) {
                Value prod = emitArithMulOp(resTy.getElementType(), b, nl, args[0], args[1]);
                b.create<linalg::YieldOp>(nl, prod);
            });
        rw.replaceOp(op, generic.getResults());
        return success();
    };

  private:
    Value emitArithMulOp(Type elemTy, OpBuilder &b, Location loc, Value lhs, Value rhs) const {
        if (isa<FloatType>(elemTy))
            return b.create<arith::MulFOp>(loc, lhs, rhs);
        return b.create<arith::MulIOp>(loc, lhs, rhs);
    }

    std::pair<Value, AffineMap> prepareRhs(daphne::EwMulOp op, RankedTensorType lhsTy, RankedTensorType resTy,
                                           AffineMap idMap, Value rhsInput, ConversionPatternRewriter &rw,
                                           Location loc) const {
        if (auto rhsTy = dyn_cast<RankedTensorType>(rhsInput.getType()))
            return std::make_pair(rhsInput, idMap);

        auto scalarTensorTy = RankedTensorType::get({}, resTy.getElementType());
        Value rhs = rw.create<tensor::FromElementsOp>(loc, scalarTensorTy, rhsInput);
        AffineMap rhsMap = AffineMap::get(resTy.getRank(), /*symbolCount=*/0, {}, rw.getContext());
        return std::make_pair(rhs, rhsMap);
    }
};

template <typename DaphneOp> struct AggAllReduce : OpConversionPattern<DaphneOp> {
    using OpConversionPattern<DaphneOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(DaphneOp op, typename DaphneOp::Adaptor ad,
                                  ConversionPatternRewriter &rw) const override {
        auto tensorTy = dyn_cast<RankedTensorType>(ad.getArg().getType());
        if (!tensorTy)
            return rw.notifyMatchFailure(op, "expected Tensor");
        auto matTy = dyn_cast<daphne::MatrixType>(op.getArg().getType());
        if (!matTy)
            return rw.notifyMatchFailure(op, "expected MatrixType");
        Type logicalTy = matTy.getElementType();
        Type storageTy = tensorTy.getElementType();
        Location loc = op.getLoc();

        Value initValue = AggReductions<DaphneOp>::identity(logicalTy, loc, rw);
        Value initT = rw.create<tensor::FromElementsOp>(loc, RankedTensorType::get({}, storageTy), initValue);

        SmallVector<int64_t> dims(tensorTy.getRank());
        std::iota(dims.begin(), dims.end(), 0);
        auto red =
            rw.create<linalg::ReduceOp>(loc, ad.getArg(), initT, dims, [&](OpBuilder &b, Location nl, ValueRange args) {
                Value comb = AggReductions<DaphneOp>::combine(logicalTy, args[0], args[1], nl, b);
                b.create<linalg::YieldOp>(nl, comb);
            });

        Value res = rw.create<tensor::ExtractOp>(loc, red.getResult(0));
        rw.replaceOp(op, res);
        return success();
    }
};

struct FillOpConverter : public OpConversionPattern<daphne::FillOp> {
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(daphne::FillOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto mt = llvm::dyn_cast<daphne::MatrixType>(op.getType());
        if (!mt)
            return rewriter.notifyMatchFailure(op, "expected MatrixType result");

        auto tensorTy = llvm::dyn_cast<RankedTensorType>(typeConverter->convertType(mt));
        if (!tensorTy)
            return rewriter.notifyMatchFailure(op, "no memref type after conversion");

        SmallVector<Value, 2> dynSizes;
        if (tensorTy.isDynamicDim(0))
            dynSizes.push_back(adaptor.getNumRows());
        if (tensorTy.isDynamicDim(1))
            dynSizes.push_back(adaptor.getNumCols());

        Location loc = op.getLoc();

        Value init = rewriter.create<tensor::EmptyOp>(loc, tensorTy, dynSizes);
        Value filled = rewriter.create<linalg::FillOp>(loc, adaptor.getArg(), init).getResult(0);

        rewriter.replaceOp(op, filled);
        return success();
    }
};

// TODO: This is temporarily duplicated from LowerToLLVMPass.
struct ReturnOpLowering : public OpRewritePattern<daphne::ReturnOp> {
    using OpRewritePattern<daphne::ReturnOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(daphne::ReturnOp op, PatternRewriter &rewriter) const final {
        rewriter.replaceOpWithNewOp<func::ReturnOp>(op, op.getOperands());
        return success();
    }
};

void populateDaphneToLinalgPatterns(DaphneTypeConverter &converter, RewritePatternSet &patterns) {
    // clang-format off
    patterns.add<
        EwMulOpConverter,
        FillOpConverter,
        AggAllReduce<daphne::AllAggSumOp>,
        AggAllReduce<daphne::AllAggMinOp>,
        AggAllReduce<daphne::AllAggMaxOp>
    >(converter, patterns.getContext());
    patterns.add<ReturnOpLowering>(patterns.getContext());
    // clang-format on
}
