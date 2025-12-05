#pragma once

#include "ir/daphneir/Daphne.h"

#include <mlir/Dialect/Arith/IR/Arith.h>

using namespace mlir;

template <typename Op> struct AggReductions;

template <> struct AggReductions<daphne::AllAggSumOp> {
    static Value identity(Type elemTy, Location loc, PatternRewriter &rw) {
        if (auto ft = dyn_cast<FloatType>(elemTy))
            return rw.create<arith::ConstantOp>(loc, ft, rw.getZeroAttr(ft));

        auto logicalTy = cast<IntegerType>(elemTy);
        auto storageTy = rw.getIntegerType(logicalTy.getWidth());
        return rw.create<arith::ConstantOp>(loc, storageTy, rw.getIntegerAttr(storageTy, 0));
    }
    static Value combine(Type elemTy, Value acc, Value elem, Location loc, OpBuilder &b) {
        if (isa<FloatType>(elemTy))
            return b.create<arith::AddFOp>(loc, acc, elem);
        return b.create<arith::AddIOp>(loc, acc, elem);
    }
};

template <> struct AggReductions<daphne::AllAggMinOp> {
    static Value identity(Type elemTy, Location loc, PatternRewriter &rw) {
        if (auto ft = dyn_cast<FloatType>(elemTy)) {
            APFloat inf = APFloat::getInf(ft.getFloatSemantics());
            return rw.create<arith::ConstantOp>(loc, ft, rw.getFloatAttr(ft, inf));
        }

        auto logicalTy = cast<IntegerType>(elemTy);
        auto storageTy = rw.getIntegerType(logicalTy.getWidth());
        APInt maxVal = logicalTy.isUnsigned() ? APInt::getMaxValue(logicalTy.getWidth())
                                              : APInt::getSignedMaxValue(logicalTy.getWidth());
        return rw.create<arith::ConstantOp>(loc, storageTy, rw.getIntegerAttr(storageTy, maxVal));
    }

    static Value combine(Type elemTy, Value acc, Value elem, Location loc, OpBuilder &b) {
        if (auto it = dyn_cast<IntegerType>(elemTy))
            return it.isUnsigned() ? b.create<arith::MinUIOp>(loc, acc, elem).getResult()
                                   : b.create<arith::MinSIOp>(loc, acc, elem).getResult();
        return b.create<arith::MinimumFOp>(loc, acc, elem);
    }
};

template <> struct AggReductions<daphne::AllAggMaxOp> {
    static Value identity(Type elemTy, Location loc, PatternRewriter &rw) {
        if (auto ft = dyn_cast<FloatType>(elemTy)) {
            APFloat negInf = APFloat::getInf(ft.getFloatSemantics(), /*Negative=*/true);
            return rw.create<arith::ConstantOp>(loc, ft, rw.getFloatAttr(ft, negInf));
        }

        auto logicalTy = cast<IntegerType>(elemTy);
        auto storageTy = rw.getIntegerType(logicalTy.getWidth());
        APInt minVal = logicalTy.isUnsigned() ? APInt::getMinValue(logicalTy.getWidth())
                                              : APInt::getSignedMinValue(logicalTy.getWidth());
        return rw.create<arith::ConstantOp>(loc, storageTy, rw.getIntegerAttr(storageTy, minVal));
    }

    static Value combine(Type elemTy, Value acc, Value elem, Location loc, OpBuilder &b) {
        if (auto it = dyn_cast<IntegerType>(elemTy))
            return it.isUnsigned() ? b.create<arith::MaxUIOp>(loc, acc, elem).getResult()
                                   : b.create<arith::MaxSIOp>(loc, acc, elem).getResult();
        return b.create<arith::MaximumFOp>(loc, acc, elem);
    }
};
