#include "compiler/conversion/DaphneToLinalg.h"
#include "compiler/conversion/AggReductions.h"
#include "ir/daphneir/Daphne.h"

#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/TypeSwitch.h>

using namespace mlir;

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

void populateDaphneToLinalgPatterns(DaphneTypeConverter &converter, RewritePatternSet &patterns) {
    // clang-format off
    patterns.add<
        FillOpConverter,
        AggAllReduce<daphne::AllAggSumOp>,
        AggAllReduce<daphne::AllAggMinOp>,
        AggAllReduce<daphne::AllAggMaxOp>
    >(converter, patterns.getContext());
    // clang-format on
}
