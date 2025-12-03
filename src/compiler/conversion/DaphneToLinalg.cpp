#include "compiler/conversion/DaphneToLinalg.h"
#include "ir/daphneir/Daphne.h"
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/TypeSwitch.h>
using namespace mlir;

template <typename Op> struct AggReductions;

template <> struct AggReductions<daphne::AllAggSumOp> {
    static Value identity(Type t, Location loc, PatternRewriter &rw) {
        return TypeSwitch<Type, Value>(t)
            .Case<FloatType>([&](FloatType ft) { return rw.create<arith::ConstantOp>(loc, ft, rw.getZeroAttr(ft)); })
            .Case<IntegerType>(
                [&](IntegerType it) { return rw.create<arith::ConstantOp>(loc, it, rw.getIntegerAttr(it, 0)); });
    }
    static Value combine(Type t, Value acc, Value elem, Location loc, OpBuilder &b) {
        if (isa<FloatType>(t))
            return b.create<arith::AddFOp>(loc, acc, elem);
        return b.create<arith::AddIOp>(loc, acc, elem);
    }
};

template <> struct AggReductions<daphne::AllAggMinOp> {
    static Value identity(Type t, Location loc, PatternRewriter &rw) {
        return TypeSwitch<Type, Value>(t)
            .Case<FloatType>([&](FloatType ft) {
                APFloat inf = APFloat::getInf(ft.getFloatSemantics());
                return rw.create<arith::ConstantOp>(loc, ft, rw.getFloatAttr(ft, inf));
            })
            .Case<IntegerType>([&](IntegerType it) {
                APInt maxVal =
                    it.isUnsigned() ? APInt::getMaxValue(it.getWidth()) : APInt::getSignedMaxValue(it.getWidth());
                return rw.create<arith::ConstantOp>(loc, it, rw.getIntegerAttr(it, maxVal));
            })
            .Default([](Type) { return nullptr; });
    }

    static Value combine(Type t, Value acc, Value elem, Location loc, OpBuilder &b) {
        if (auto it = dyn_cast<IntegerType>(t))
            return it.isUnsigned() ? b.create<arith::MinUIOp>(loc, acc, elem).getResult()
                                   : b.create<arith::MinSIOp>(loc, acc, elem).getResult();
        return b.create<arith::MinimumFOp>(loc, acc, elem);
    }
};

template <> struct AggReductions<daphne::AllAggMaxOp> {
    static Value identity(Type t, Location loc, PatternRewriter &rw) {
        return TypeSwitch<Type, Value>(t)
            .Case<FloatType>([&](FloatType ft) {
                APFloat negInf = APFloat::getInf(ft.getFloatSemantics(), /*Negative=*/true);
                return rw.create<arith::ConstantOp>(loc, ft, rw.getFloatAttr(ft, negInf));
            })
            .Case<IntegerType>([&](IntegerType it) {
                APInt minVal =
                    it.isUnsigned() ? APInt::getMinValue(it.getWidth()) : APInt::getSignedMinValue(it.getWidth());
                return rw.create<arith::ConstantOp>(loc, it, rw.getIntegerAttr(it, minVal));
            })
            .Default([](Type) { return nullptr; });
    }

    static Value combine(Type t, Value acc, Value elem, Location loc, OpBuilder &b) {
        if (auto it = dyn_cast<IntegerType>(t))
            return it.isUnsigned() ? b.create<arith::MaxUIOp>(loc, acc, elem).getResult()
                                   : b.create<arith::MaxSIOp>(loc, acc, elem).getResult();
        return b.create<arith::MaximumFOp>(loc, acc, elem);
    }
};

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
    patterns.add<FillOpConverter, AggAllReduce<daphne::AllAggSumOp>, AggAllReduce<daphne::AllAggMinOp>,
                 AggAllReduce<daphne::AllAggMaxOp>>(converter, patterns.getContext());
}
