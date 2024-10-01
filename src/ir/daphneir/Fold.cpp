#include "mlir/IR/OpDefinition.h"
#include <ir/daphneir/Daphne.h>
#include <ir/daphneir/DaphneOps.cpp.inc>
#include <util/ErrorHandler.h>

mlir::Attribute performCast(mlir::Attribute attr, mlir::Type targetType, mlir::Location loc) {
    if (auto intAttr = attr.dyn_cast<mlir::IntegerAttr>()) {
        auto apInt = intAttr.getValue();

        if (auto outTy = targetType.dyn_cast<mlir::IntegerType>()) {
            // Extend or truncate the integer value based on the target type
            if (outTy.isUnsignedInteger()) {
                apInt = apInt.zextOrTrunc(outTy.getWidth());
            } else if (outTy.isSignedInteger()) {
                apInt = (intAttr.getType().isSignedInteger()) ? apInt.sextOrTrunc(outTy.getWidth())
                                                              : apInt.zextOrTrunc(outTy.getWidth());
            }
            return mlir::IntegerAttr::getChecked(loc, outTy, apInt);
        }

        if (auto outTy = targetType.dyn_cast<mlir::IndexType>()) {
            return mlir::IntegerAttr::getChecked(loc, outTy, apInt);
        }

        if (targetType.isF64()) {
            if (intAttr.getType().isSignedInteger()) {
                return mlir::FloatAttr::getChecked(loc, targetType, llvm::APIntOps::RoundSignedAPIntToDouble(apInt));
            }
            if (intAttr.getType().isUnsignedInteger() || intAttr.getType().isIndex()) {
                return mlir::FloatAttr::getChecked(loc, targetType, llvm::APIntOps::RoundAPIntToDouble(apInt));
            }
        }

        if (targetType.isF32()) {
            if (intAttr.getType().isSignedInteger()) {
                return mlir::FloatAttr::getChecked(loc, targetType, llvm::APIntOps::RoundSignedAPIntToFloat(apInt));
            }
            if (intAttr.getType().isUnsignedInteger()) {
                return mlir::FloatAttr::get(targetType, llvm::APIntOps::RoundAPIntToFloat(apInt));
            }
        }
    } else if (auto floatAttr = attr.dyn_cast<mlir::FloatAttr>()) {
        auto val = floatAttr.getValueAsDouble();

        if (targetType.isF64()) {
            return mlir::FloatAttr::getChecked(loc, targetType, val);
        }
        if (targetType.isF32()) {
            return mlir::FloatAttr::getChecked(loc, targetType, static_cast<float>(val));
        }
        if (targetType.isIntOrIndex()) {
            auto num = static_cast<int64_t>(val);
            return mlir::IntegerAttr::getChecked(loc, targetType, num);
        }
    }

    // If casting is not possible, return the original attribute
    return {};
}

template <class AttrElementT, class ElementValueT = typename AttrElementT::ValueType,
          class CalculationT = std::function<ElementValueT(const ElementValueT &)>>
mlir::Attribute constFoldUnaryOp(mlir::Location loc, mlir::Type resultType, llvm::ArrayRef<mlir::Attribute> operands,
                                 const CalculationT &calculate) {
    if (operands.size() != 1)
        throw ErrorHandler::compilerError(loc, "CanonicalizerPass (constFoldUnaryOp)",
                                          "unary op takes one operand but " + std::to_string(operands.size()) +
                                              " were given");

    if (!operands[0])
        return {};

    if (llvm::isa<AttrElementT>(operands[0])) {
        auto operand = operands[0].cast<AttrElementT>();

        return AttrElementT::get(resultType, calculate(operand.getValue()));
    }
    return {};
}

template <class ArgAttrElementT, class ResAttrElementT = ArgAttrElementT,
          class ArgElementValueT = typename ArgAttrElementT::ValueType,
          class ResElementValueT = typename ResAttrElementT::ValueType,
          class CalculationT = std::function<ResElementValueT(const ArgElementValueT &, const ArgElementValueT &)>>
mlir::Attribute constFoldBinaryOp(mlir::Location loc, mlir::Type resultType, llvm::ArrayRef<mlir::Attribute> operands,
                                  const CalculationT &calculate) {
    if (operands.size() != 2)
        throw ErrorHandler::compilerError(loc, "CanonicalizerPass (constFoldBinaryOp)",
                                          "binary op takes two operands but " + std::to_string(operands.size()) +
                                              " were given");

    if (!operands[0] || !operands[1])
        return {};

    if (llvm::isa<ArgAttrElementT>(operands[0]) && llvm::isa<ArgAttrElementT>(operands[1])) {
        auto lhs = operands[0].cast<ArgAttrElementT>();
        auto rhs = operands[1].cast<ArgAttrElementT>();

        // We need dedicated cases, as the parameters of ResAttrElementT::get()
        // depend on ResAttrElementT.
        if constexpr (std::is_same<ResAttrElementT, mlir::IntegerAttr>::value ||
                      std::is_same<ResAttrElementT, mlir::FloatAttr>::value) {
            mlir::Type l = lhs.getType();
            mlir::Type r = rhs.getType();
            if ((l.dyn_cast<mlir::IntegerType>() || l.dyn_cast<mlir::FloatType>()) &&
                (r.dyn_cast<mlir::IntegerType>() || r.dyn_cast<mlir::FloatType>())) {
                auto lhsBitWidth = lhs.getType().getIntOrFloatBitWidth();
                auto rhsBitWidth = rhs.getType().getIntOrFloatBitWidth();

                if (lhsBitWidth < rhsBitWidth) {
                    mlir::Attribute promotedLhs = performCast(lhs, rhs.getType(), loc);
                    lhs = promotedLhs.cast<ArgAttrElementT>();
                } else if (rhsBitWidth < lhsBitWidth) {
                    mlir::Attribute promotedRhs = performCast(rhs, lhs.getType(), loc);
                    rhs = promotedRhs.cast<ArgAttrElementT>();
                }
            }
            return ResAttrElementT::get(resultType, calculate(lhs.getValue(), rhs.getValue()));
        } else if constexpr (std::is_same<ResAttrElementT, mlir::BoolAttr>::value) {
            if (!resultType.isSignlessInteger(1))
                throw ErrorHandler::compilerError(loc, "CanonicalizerPass (constFoldBinaryOp)",
                                                  "expected boolean result type");
            return ResAttrElementT::get(lhs.getContext(), calculate(lhs.getValue(), rhs.getValue()));
        } else if constexpr (std::is_same<ResAttrElementT, mlir::StringAttr>::value) {
            if (!resultType.isa<mlir::daphne::StringType>())
                throw ErrorHandler::compilerError(loc, "CanonicalizerPass (constFoldBinaryOp)",
                                                  "expected string result type");
            return ResAttrElementT::get(calculate(lhs.getValue(), rhs.getValue()), resultType);
        }
    }
    return {};
}

mlir::OpFoldResult mlir::daphne::ConstantOp::fold(FoldAdaptor adaptor) {
    if (!adaptor.getOperands().empty())
        throw ErrorHandler::compilerError(this->getLoc(), "CanonicalizerPass (mlir::daphne::ConstantOp::fold)",
                                          "constant has no operands but " +
                                              std::to_string(adaptor.getOperands().size()) + " were given");

    return getValue();
}

mlir::OpFoldResult mlir::daphne::CastOp::fold(FoldAdaptor adaptor) {
    ArrayRef<Attribute> operands = adaptor.getOperands();

    if (isTrivialCast()) {
        if (operands[0])
            return {operands[0]};
        else
            return {getArg()};
    }

    if (operands[0]) {
        if (auto castedAttr = performCast(operands[0], getType(), getLoc())) {
            return castedAttr;
        }
    }

    return {};
}

mlir::OpFoldResult mlir::daphne::EwAddOp::fold(FoldAdaptor adaptor) {
    ArrayRef<Attribute> operands = adaptor.getOperands();
    auto floatOp = [](const llvm::APFloat &a, const llvm::APFloat &b) { return a + b; };
    // TODO: we could check overflows
    auto intOp = [](const llvm::APInt &a, const llvm::APInt &b) { return a + b; };
    if (auto res = constFoldBinaryOp<FloatAttr>(getLoc(), getType(), operands, floatOp))
        return res;
    if (auto res = constFoldBinaryOp<IntegerAttr>(getLoc(), getType(), operands, intOp))
        return res;
    return {};
}

mlir::OpFoldResult mlir::daphne::EwSubOp::fold(FoldAdaptor adaptor) {
    ArrayRef<Attribute> operands = adaptor.getOperands();
    auto floatOp = [](const llvm::APFloat &a, const llvm::APFloat &b) { return a - b; };
    auto intOp = [](const llvm::APInt &a, const llvm::APInt &b) { return a - b; };
    if (auto res = constFoldBinaryOp<FloatAttr>(getLoc(), getType(), operands, floatOp))
        return res;
    if (auto res = constFoldBinaryOp<IntegerAttr>(getLoc(), getType(), operands, intOp))
        return res;
    return {};
}

mlir::OpFoldResult mlir::daphne::EwMulOp::fold(FoldAdaptor adaptor) {
    ArrayRef<Attribute> operands = adaptor.getOperands();
    auto floatOp = [](const llvm::APFloat &a, const llvm::APFloat &b) { return a * b; };
    auto intOp = [](const llvm::APInt &a, const llvm::APInt &b) { return a * b; };
    if (auto res = constFoldBinaryOp<FloatAttr>(getLoc(), getType(), operands, floatOp))
        return res;
    if (auto res = constFoldBinaryOp<IntegerAttr>(getLoc(), getType(), operands, intOp))
        return res;
    return {};
}

mlir::OpFoldResult mlir::daphne::EwDivOp::fold(FoldAdaptor adaptor) {
    ArrayRef<Attribute> operands = adaptor.getOperands();
    auto floatOp = [](const llvm::APFloat &a, const llvm::APFloat &b) { return a / b; };
    auto sintOp = [&](const llvm::APInt &a, const llvm::APInt &b) {
        if (b == 0) {
            throw ErrorHandler::compilerError(this->getLoc(), "CanonicalizerPass (mlir::daphne::EwDivOp::fold)",
                                              "Can't divide by 0");
        }
        return a.sdiv(b);
    };
    auto uintOp = [&](const llvm::APInt &a, const llvm::APInt &b) {
        if (b == 0) {
            throw ErrorHandler::compilerError(this->getLoc(), "CanonicalizerPass (mlir::daphne::EwDivOp::fold)",
                                              "Can't divide by 0");
        }
        return a.udiv(b);
    };

    if (auto res = constFoldBinaryOp<FloatAttr>(getLoc(), getType(), operands, floatOp))
        return res;
    if (getType().isSignedInteger()) {
        if (auto res = constFoldBinaryOp<IntegerAttr>(getLoc(), getType(), operands, sintOp))
            return res;
    } else if (getType().isUnsignedInteger()) {
        if (auto res = constFoldBinaryOp<IntegerAttr>(getLoc(), getType(), operands, uintOp))
            return res;
    }
    return {};
}

mlir::OpFoldResult mlir::daphne::EwMinusOp::fold(FoldAdaptor adaptor) {
    ArrayRef<Attribute> operands = adaptor.getOperands();
    auto intOp = [](const llvm::APInt &a) { return -a; };
    auto floatOp = [](const llvm::APFloat &a) { return -a; };

    if (auto res = constFoldUnaryOp<IntegerAttr>(getLoc(), getType(), operands, intOp))
        return res;
    if (auto res = constFoldUnaryOp<FloatAttr>(getLoc(), getType(), operands, floatOp))
        return res;

    return {};
}

mlir::OpFoldResult mlir::daphne::EwPowOp::fold(FoldAdaptor adaptor) {
    ArrayRef<Attribute> operands = adaptor.getOperands();
    // TODO: EwPowOp integer constant folding
    auto floatOp = [](const llvm::APFloat &a, const llvm::APFloat &b) {
        return std::pow(a.convertToDouble(), b.convertToDouble());
    };
    if (auto res = constFoldBinaryOp<FloatAttr>(getLoc(), getType(), operands, floatOp))
        return res;
    return {};
}

mlir::OpFoldResult mlir::daphne::EwModOp::fold(FoldAdaptor adaptor) {
    ArrayRef<Attribute> operands = adaptor.getOperands();
    auto sintOp = [&](const llvm::APInt &a, const llvm::APInt &b) {
        if (b == 0) {
            throw ErrorHandler::compilerError(this->getLoc(), "CanonicalizerPass (mlir::daphne::EwModOp::fold)",
                                              "Can't compute mod 0");
        }
        return a.srem(b);
    };
    auto uintOp = [&](const llvm::APInt &a, const llvm::APInt &b) {
        if (b == 0) {
            throw ErrorHandler::compilerError(this->getLoc(), "CanonicalizerPass (mlir::daphne::EwModOp::fold)",
                                              "Can't compute mod 0");
        }
        return a.urem(b);
    };
    if (getType().isSignedInteger()) {
        if (auto res = constFoldBinaryOp<IntegerAttr>(getLoc(), getType(), operands, sintOp))
            return res;
    } else if (getType().isUnsignedInteger()) {
        if (auto res = constFoldBinaryOp<IntegerAttr>(getLoc(), getType(), operands, uintOp))
            return res;
    }
    return {};
}

mlir::OpFoldResult mlir::daphne::EwLogOp::fold(FoldAdaptor adaptor) {
    ArrayRef<Attribute> operands = adaptor.getOperands();
    auto floatOp = [](const llvm::APFloat &a, const llvm::APFloat &b) {
        // Compute the element-wise logarithm of a to the base b
        // Equivalent to log_b(a)
        return log(a.convertToDouble()) / log(b.convertToDouble());
    };
    if (auto res = constFoldBinaryOp<FloatAttr>(getLoc(), getType(), operands, floatOp))
        return res;
    return {};
}

mlir::OpFoldResult mlir::daphne::EwMinOp::fold(FoldAdaptor adaptor) {
    ArrayRef<Attribute> operands = adaptor.getOperands();
    auto floatOp = [](const llvm::APFloat &a, const llvm::APFloat &b) { return llvm::minimum(a, b); };
    auto sintOp = [&](const llvm::APInt &a, const llvm::APInt &b) {
        if (a.slt(b))
            return a;
        else
            return b;
    };
    auto uintOp = [&](const llvm::APInt &a, const llvm::APInt &b) {
        if (a.ult(b))
            return a;
        else
            return b;
    };
    if (auto res = constFoldBinaryOp<FloatAttr>(getLoc(), getType(), operands, floatOp))
        return res;
    if (getType().isSignedInteger()) {
        if (auto res = constFoldBinaryOp<IntegerAttr>(getLoc(), getType(), operands, sintOp))
            return res;
    } else if (getType().isUnsignedInteger()) {
        if (auto res = constFoldBinaryOp<IntegerAttr>(getLoc(), getType(), operands, uintOp))
            return res;
    }
    return {};
}

mlir::OpFoldResult mlir::daphne::EwMaxOp::fold(FoldAdaptor adaptor) {
    ArrayRef<Attribute> operands = adaptor.getOperands();
    auto floatOp = [](const llvm::APFloat &a, const llvm::APFloat &b) { return llvm::maximum(a, b); };
    auto sintOp = [&](const llvm::APInt &a, const llvm::APInt &b) {
        if (a.sgt(b))
            return a;
        else
            return b;
    };
    auto uintOp = [&](const llvm::APInt &a, const llvm::APInt &b) {
        if (a.ugt(b))
            return a;
        else
            return b;
    };
    if (auto res = constFoldBinaryOp<FloatAttr>(getLoc(), getType(), operands, floatOp))
        return res;
    if (getType().isSignedInteger()) {
        if (auto res = constFoldBinaryOp<IntegerAttr>(getLoc(), getType(), operands, sintOp))
            return res;
    } else if (getType().isUnsignedInteger()) {
        if (auto res = constFoldBinaryOp<IntegerAttr>(getLoc(), getType(), operands, uintOp))
            return res;
    }
    return {};
}

mlir::OpFoldResult mlir::daphne::EwAndOp::fold(FoldAdaptor adaptor) {
    ArrayRef<Attribute> operands = adaptor.getOperands();
    auto boolOp = [](const bool &a, const bool &b) { return a && b; };
    auto intOp = [](const llvm::APInt &a, const llvm::APInt &b) { return (a != 0) && (b != 0); };
    if (auto res = constFoldBinaryOp<BoolAttr>(getLoc(), getType(), operands, boolOp))
        return res;
    // TODO: should output bool?
    if (auto res = constFoldBinaryOp<IntegerAttr>(getLoc(), getType(), operands, intOp))
        return res;
    return {};
}

mlir::OpFoldResult mlir::daphne::EwBitwiseAndOp::fold(FoldAdaptor adaptor) { return {}; }

mlir::OpFoldResult mlir::daphne::EwOrOp::fold(FoldAdaptor adaptor) {
    ArrayRef<Attribute> operands = adaptor.getOperands();
    auto boolOp = [](const bool &a, const bool &b) { return a || b; };
    auto intOp = [](const llvm::APInt &a, const llvm::APInt &b) { return (a != 0) || (b != 0); };
    if (auto res = constFoldBinaryOp<BoolAttr>(getLoc(), getType(), operands, boolOp))
        return res;
    // TODO: should output bool
    if (auto res = constFoldBinaryOp<IntegerAttr>(getLoc(), getType(), operands, intOp))
        return res;
    return {};
}

mlir::OpFoldResult mlir::daphne::EwXorOp::fold(FoldAdaptor adaptor) {
    ArrayRef<Attribute> operands = adaptor.getOperands();
    auto boolOp = [](const bool &a, const bool &b) { return a ^ b; };
    auto intOp = [](const llvm::APInt &a, const llvm::APInt &b) { return (a != 0) ^ (b != 0); };
    if (auto res = constFoldBinaryOp<BoolAttr>(getLoc(), getType(), operands, boolOp))
        return res;
    // TODO: should output bool
    if (auto res = constFoldBinaryOp<IntegerAttr>(getLoc(), getType(), operands, intOp))
        return res;
    return {};
}

mlir::OpFoldResult mlir::daphne::EwConcatOp::fold(FoldAdaptor adaptor) {
    ArrayRef<Attribute> operands = adaptor.getOperands();

    if (operands.size() != 2)
        throw ErrorHandler::compilerError(this->getLoc(), "CanonicalizerPass (mlir::daphne::EwConcatOp::fold)",
                                          "binary op takes two operands but " + std::to_string(operands.size()) +
                                              " were given");

    if (!operands[0] || !operands[1])
        return {};

    if (llvm::isa<StringAttr>(operands[0]) && isa<StringAttr>(operands[1])) {
        auto lhs = operands[0].cast<StringAttr>();
        auto rhs = operands[1].cast<StringAttr>();

        auto concated = lhs.getValue().str() + rhs.getValue().str();
        return StringAttr::get(concated, getType());
    }
    return {};
}

mlir::OpFoldResult mlir::daphne::EwEqOp::fold(FoldAdaptor adaptor) {
    ArrayRef<Attribute> operands = adaptor.getOperands();
    auto floatOp = [](const llvm::APFloat &a, const llvm::APFloat &b) { return a == b; };
    auto intOp = [](const llvm::APInt &a, const llvm::APInt &b) { return a == b; };
    auto strOp = [](const llvm::StringRef &a, const llvm::StringRef &b) { return a == b; };
    // TODO: fix bool return
    if (auto res = constFoldBinaryOp<FloatAttr>(getLoc(), getType(), operands, floatOp))
        return res;
    if (auto res = constFoldBinaryOp<IntegerAttr>(getLoc(), getType(), operands, intOp))
        return res;
    if (auto res = constFoldBinaryOp<StringAttr, IntegerAttr>(
            getLoc(), IntegerType::get(getContext(), 64, IntegerType::SignednessSemantics::Signed), operands, strOp))
        return res;
    return {};
}

mlir::OpFoldResult mlir::daphne::EwNeqOp::fold(FoldAdaptor adaptor) {
    ArrayRef<Attribute> operands = adaptor.getOperands();
    auto floatOp = [](const llvm::APFloat &a, const llvm::APFloat &b) { return a != b; };
    auto intOp = [](const llvm::APInt &a, const llvm::APInt &b) { return a != b; };
    // TODO: fix bool return
    if (auto res = constFoldBinaryOp<FloatAttr>(getLoc(), getType(), operands, floatOp))
        return res;
    if (auto res = constFoldBinaryOp<IntegerAttr>(getLoc(), getType(), operands, intOp))
        return res;
    return {};
}

mlir::OpFoldResult mlir::daphne::EwLtOp::fold(FoldAdaptor adaptor) {
    ArrayRef<Attribute> operands = adaptor.getOperands();
    auto floatOp = [](const llvm::APFloat &a, const llvm::APFloat &b) { return a < b; };
    auto sintOp = [](const llvm::APInt &a, const llvm::APInt &b) { return a.slt(b); };
    auto uintOp = [](const llvm::APInt &a, const llvm::APInt &b) { return a.ult(b); };
    // TODO: fix bool return
    if (auto res = constFoldBinaryOp<FloatAttr>(getLoc(), getType(), operands, floatOp))
        return res;
    if (getType().isSignedInteger()) {
        if (auto res = constFoldBinaryOp<IntegerAttr>(getLoc(), getType(), operands, sintOp))
            return res;
    } else if (getType().isUnsignedInteger()) {
        if (auto res = constFoldBinaryOp<IntegerAttr>(getLoc(), getType(), operands, uintOp))
            return res;
    }
    return {};
}

mlir::OpFoldResult mlir::daphne::EwLeOp::fold(FoldAdaptor adaptor) {
    ArrayRef<Attribute> operands = adaptor.getOperands();
    auto floatOp = [](const llvm::APFloat &a, const llvm::APFloat &b) { return a <= b; };
    auto sintOp = [](const llvm::APInt &a, const llvm::APInt &b) { return a.sle(b); };
    auto uintOp = [](const llvm::APInt &a, const llvm::APInt &b) { return a.ule(b); };
    // TODO: fix bool return
    if (auto res = constFoldBinaryOp<FloatAttr>(getLoc(), getType(), operands, floatOp))
        return res;
    if (getType().isSignedInteger()) {
        if (auto res = constFoldBinaryOp<IntegerAttr>(getLoc(), getType(), operands, sintOp))
            return res;
    } else if (getType().isUnsignedInteger()) {
        if (auto res = constFoldBinaryOp<IntegerAttr>(getLoc(), getType(), operands, uintOp))
            return res;
    }
    return {};
}

mlir::OpFoldResult mlir::daphne::EwGtOp::fold(FoldAdaptor adaptor) {
    ArrayRef<Attribute> operands = adaptor.getOperands();
    auto floatOp = [](const llvm::APFloat &a, const llvm::APFloat &b) { return a > b; };
    auto sintOp = [](const llvm::APInt &a, const llvm::APInt &b) { return a.sgt(b); };
    auto uintOp = [](const llvm::APInt &a, const llvm::APInt &b) { return a.ugt(b); };
    // TODO: fix bool return
    if (auto res = constFoldBinaryOp<FloatAttr>(getLoc(), getType(), operands, floatOp))
        return res;
    if (getType().isSignedInteger()) {
        if (auto res = constFoldBinaryOp<IntegerAttr>(getLoc(), getType(), operands, sintOp))
            return res;
    } else if (getType().isUnsignedInteger()) {
        if (auto res = constFoldBinaryOp<IntegerAttr>(getLoc(), getType(), operands, uintOp))
            return res;
    }
    return {};
}

mlir::OpFoldResult mlir::daphne::EwGeOp::fold(FoldAdaptor adaptor) {
    ArrayRef<Attribute> operands = adaptor.getOperands();
    auto floatOp = [](const llvm::APFloat &a, const llvm::APFloat &b) { return a >= b; };
    auto sintOp = [](const llvm::APInt &a, const llvm::APInt &b) { return a.sge(b); };
    auto uintOp = [](const llvm::APInt &a, const llvm::APInt &b) { return a.uge(b); };
    // TODO: fix bool return
    if (auto res = constFoldBinaryOp<FloatAttr>(getLoc(), getType(), operands, floatOp))
        return res;
    if (getType().isSignedInteger()) {
        if (auto res = constFoldBinaryOp<IntegerAttr>(getLoc(), getType(), operands, sintOp))
            return res;
    } else if (getType().isUnsignedInteger()) {
        if (auto res = constFoldBinaryOp<IntegerAttr>(getLoc(), getType(), operands, uintOp))
            return res;
    }
    return {};
}
