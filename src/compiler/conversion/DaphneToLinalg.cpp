#include "compiler/conversion/DaphneToLinalg.h"
#include "compiler/conversion/AggReductions.h"
#include "ir/daphneir/Daphne.h"

#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/TypeSwitch.h>

template <typename DaphneOp> struct AggAllReduce : OpConversionPattern<DaphneOp> {
    using OpConversionPattern<DaphneOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(DaphneOp op, typename DaphneOp::Adaptor ad,
                                  ConversionPatternRewriter &rw) const override {
        auto memrefTy = dyn_cast<MemRefType>(ad.getArg().getType());
        if (!memrefTy)
            return rw.notifyMatchFailure(op, "expected memref");
        auto matTy = dyn_cast<daphne::MatrixType>(op.getArg().getType());
        if (!matTy)
            return rw.notifyMatchFailure(op, "expected MatrixType");
        Type logicalTy = matTy.getElementType();
        Type storageTy = memrefTy.getElementType();
        Location loc = op.getLoc();

        auto tensorTy = RankedTensorType::get(memrefTy.getShape(), storageTy);
        Value tin = rw.create<bufferization::ToTensorOp>(loc, tensorTy, ad.getArg(), /*restricted=*/true);
        Value initValue = AggReductions<DaphneOp>::identity(logicalTy, loc, rw);
        Value initT = rw.create<tensor::FromElementsOp>(loc, RankedTensorType::get({}, storageTy), initValue);

        SmallVector<int64_t> dims(memrefTy.getRank());
        std::iota(dims.begin(), dims.end(), 0);
        auto red = rw.create<linalg::ReduceOp>(loc, tin, initT, dims, [&](OpBuilder &b, Location nl, ValueRange args) {
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

        auto memrefTy = llvm::dyn_cast<MemRefType>(typeConverter->convertType(mt));
        if (!memrefTy)
            return rewriter.notifyMatchFailure(op, "no memref type after conversion");

        SmallVector<Value, 2> dynSizes;
        if (memrefTy.isDynamicDim(0))
            dynSizes.push_back(adaptor.getNumRows());
        if (memrefTy.isDynamicDim(1))
            dynSizes.push_back(adaptor.getNumCols());

        Location loc = op.getLoc();
        Value alloc = rewriter.create<memref::AllocOp>(loc, memrefTy, dynSizes);

        rewriter.create<linalg::FillOp>(loc, adaptor.getArg(), alloc);
        rewriter.replaceOp(op, alloc);
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
