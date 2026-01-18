#include "ir/daphneir/Daphne.h"

#include <mlir/Dialect/Arith/IR/Arith.h>

template <typename BinaryOp> struct EwBinaryOpTraits {};

template <> struct EwBinaryOpTraits<mlir::daphne::EwAddOp> {
    static mlir::Value emit(mlir::Type elemTy, mlir::OpBuilder &b, mlir::Location loc, mlir::Value lhs,
                            mlir::Value rhs) {
        if (isa<mlir::FloatType>(elemTy))
            return b.create<mlir::arith::AddFOp>(loc, lhs, rhs);
        return b.create<mlir::arith::AddIOp>(loc, lhs, rhs);
    }
};

template <> struct EwBinaryOpTraits<mlir::daphne::EwSubOp> {
    static mlir::Value emit(mlir::Type elemTy, mlir::OpBuilder &b, mlir::Location loc, mlir::Value lhs,
                            mlir::Value rhs) {
        if (isa<mlir::FloatType>(elemTy))
            return b.create<mlir::arith::SubFOp>(loc, lhs, rhs);
        return b.create<mlir::arith::SubIOp>(loc, lhs, rhs);
    }
};

template <> struct EwBinaryOpTraits<mlir::daphne::EwMulOp> {
    static mlir::Value emit(mlir::Type elemTy, mlir::OpBuilder &b, mlir::Location loc, mlir::Value lhs,
                            mlir::Value rhs) {
        if (isa<mlir::FloatType>(elemTy))
            return b.create<mlir::arith::MulFOp>(loc, lhs, rhs);
        return b.create<mlir::arith::MulIOp>(loc, lhs, rhs);
    }
};

template <> struct EwBinaryOpTraits<mlir::daphne::EwDivOp> {
    static mlir::Value emit(mlir::Type elemTy, mlir::OpBuilder &b, mlir::Location loc, mlir::Value lhs,
                            mlir::Value rhs) {
        if (isa<mlir::FloatType>(elemTy))
            return b.create<mlir::arith::DivFOp>(loc, lhs, rhs);
        auto it = dyn_cast<mlir::IntegerType>(elemTy);
        if (it.isUnsigned())
            return b.create<mlir::arith::DivUIOp>(loc, lhs, rhs);
        return b.create<mlir::arith::DivSIOp>(loc, lhs, rhs);
    }
};
