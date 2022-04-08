/*
 * Copyright 2021 The DAPHNE Consortium
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

#include <ir/daphneir/Daphne.h>
#include <ir/daphneir/DaphneOpsEnums.cpp.inc>
#define GET_OP_CLASSES
#include <ir/daphneir/DaphneOps.cpp.inc>
#define GET_TYPEDEF_CLASSES
#include <ir/daphneir/DaphneOpsTypes.cpp.inc>

#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/APSInt.h>

void mlir::daphne::DaphneDialect::initialize()
{
    addOperations<
        #define GET_OP_LIST
        #include <ir/daphneir/DaphneOps.cpp.inc>
    >();
    addTypes<
        #define GET_TYPEDEF_LIST
        #include <ir/daphneir/DaphneOpsTypes.cpp.inc>
    >();
}

mlir::Operation *mlir::daphne::DaphneDialect::materializeConstant(OpBuilder &builder,
                                                                  Attribute value, Type type,
                                                                  mlir::Location loc)
{
    return builder.create<mlir::daphne::ConstantOp>(loc, type, value);
}

mlir::Type mlir::daphne::DaphneDialect::parseType(mlir::DialectAsmParser &parser) const
{
    llvm::StringRef keyword;
    parser.parseKeyword(&keyword);
    // `Matrix` `<` (`?` | \d+) `x` (`?` | \d+) `x` \type
    //      (`:` (
    //          `sp` `[` \float `]` |
    //          `rep` `[` (`dense` | `sparse`) `]`
    //      ))*
    if (keyword == "Matrix") {
        ssize_t numRows = -1;
        ssize_t numCols = -1;
        double sparsity = -1.0;
        MatrixRepresentation representation = MatrixRepresentation::Default; // default is dense
        mlir::Type elementType;
        if (
            parser.parseLess() ||
            parser.parseOptionalQuestion() ||
            // TODO Parse #rows if there was no '?'.
            //parser.parseInteger<ssize_t>(numRows) ||
            parser.parseXInDimensionList() ||
            parser.parseOptionalQuestion() ||
            // TODO Parse #cols if there was no '?'.
            //parser.parseInteger<ssize_t>(numCols) ||
            parser.parseXInDimensionList() ||
            parser.parseType(elementType)
        ) {
            return nullptr;
        }
        // additional properties (only print/read them when present, as this will probably get more and more)
        while (succeeded(parser.parseOptionalColon())) {
            if (succeeded(parser.parseKeyword("sp"))) {
                if (sparsity != -1.0) {
                    // read sparsity twice
                    return nullptr;
                }
                if (parser.parseLSquare() || parser.parseFloat(sparsity) || parser.parseRSquare()) {
                    return nullptr;
                }
            }
            else if (succeeded(parser.parseKeyword("rep"))) {
                llvm::StringRef repName;
                if (parser.parseLSquare() || parser.parseKeyword(&repName) || parser.parseRSquare()) {
                    return nullptr;
                }
                representation = stringToMatrixRepresentation(repName.str());
            }
            else {
                return nullptr;
            }
        }
        if(parser.parseGreater()) {
            return nullptr;
        }

        return MatrixType::get(
                parser.getBuilder().getContext(), elementType, numRows, numCols, sparsity, representation
        );
    }
    else if (keyword == "Frame") {
        ssize_t numRows = -1;
        ssize_t numCols = -1;
        if (
            parser.parseLess() ||
            parser.parseOptionalQuestion() ||
            // TODO Parse #rows if there was no '?'.
            //parser.parseInteger<ssize_t>(numRows) ||
            parser.parseKeyword("x") ||
            parser.parseLSquare() ||
            parser.parseOptionalQuestion() ||
            // TODO Parse #cols if there was no '?'.
            //parser.parseInteger<ssize_t>(numCols) ||
            // TODO Parse sparsity
            parser.parseColon()
        ) {
            return nullptr;
        }
        std::vector<mlir::Type> cts;
        mlir::Type type;
        do {
            if (parser.parseType(type))
                return nullptr;
            cts.push_back(type);
        }
        while (succeeded(parser.parseOptionalComma()));
        if (parser.parseRSquare() || parser.parseGreater()) {
            return nullptr;
        }
        return FrameType::get(
                parser.getBuilder().getContext(), cts, numRows, numCols, nullptr
        );
    }
    else if (keyword == "Handle") {
        mlir::Type dataType;
        if (parser.parseLess() || parser.parseType(dataType) || parser.parseGreater()) {
            return nullptr;
        }
        return mlir::daphne::HandleType::get(parser.getBuilder().getContext(), dataType);
    }
    else if (keyword == "String") {
        return StringType::get(parser.getBuilder().getContext());
    }
    else {
        parser.emitError(parser.getCurrentLocation()) << "Parsing failed, keyword `" << keyword << "` not recognized!";
        return nullptr;
    }
}

std::string unknownStrIf(ssize_t val) {
    return (val == -1) ? "?" : std::to_string(val);
}

std::string unknownStrIf(double val) {
    return (val == -1.0) ? "?" : std::to_string(val);
}

void mlir::daphne::DaphneDialect::printType(mlir::Type type,
                                            mlir::DialectAsmPrinter &os) const
{
    if (auto t = type.dyn_cast<mlir::daphne::MatrixType>()) {
        os << "Matrix<"
                << unknownStrIf(t.getNumRows()) << 'x'
                << unknownStrIf(t.getNumCols()) << 'x'
                << t.getElementType();
        auto sparsity = t.getSparsity();
        auto representation = t.getRepresentation();

        if (sparsity != -1.0) {
            os << ":sp[" << sparsity << ']';
        }
        if (representation != MatrixRepresentation::Default) {
            os << ":rep[" << matrixRepresentationToString(representation) << ']';
        }
        os << '>';
    }
    else if (auto t = type.dyn_cast<mlir::daphne::FrameType>()) {
        os << "Frame<"
                << unknownStrIf(t.getNumRows()) << "x["
                << unknownStrIf(t.getNumCols()) << ": ";
        // Column types.
        std::vector<mlir::Type> cts = t.getColumnTypes();
        for (size_t i = 0; i < cts.size(); i++) {
            os << cts[i];
            if(i < cts.size() - 1)
                os << ", ";
        }
        os << "], ";
        // Column labels.
        std::vector<std::string> * labels = t.getLabels();
        if(labels) {
            os << '[';
            for (size_t i = 0; i < labels->size(); i++) {
                os << '"' << (*labels)[i] << '"';
                if(i < labels->size() - 1)
                    os << ", ";
            }
            os << ']';
        }
        else
            os << '?';
        os << '>';
    }
    else if (auto handle = type.dyn_cast<mlir::daphne::HandleType>()) {
        os << "Handle<" << handle.getDataType() << ">";
    }
    else if (type.isa<mlir::daphne::StringType>())
        os << "String";
    else if (auto t = type.dyn_cast<mlir::daphne::VariadicPackType>())
        os << "VariadicPack<" << t.getContainedType() << '>';
    else if (type.isa<mlir::daphne::DaphneContextType>())
        os << "DaphneContext";
    else if (type.isa<mlir::daphne::FileType>())
        os << "File";
    else if (type.isa<mlir::daphne::DescriptorType>())
        os << "Descriptor";
    else if (type.isa<mlir::daphne::TargetType>())
        os << "Target";
    else if (type.isa<mlir::daphne::UnknownType>())
        os << "Unknown";
}

std::string mlir::daphne::matrixRepresentationToString(MatrixRepresentation rep) {
    switch (rep) {
    case MatrixRepresentation::Dense:
        return "dense";
    case MatrixRepresentation::Sparse:
        return "sparse";
    default:
        throw std::runtime_error("unknown mlir::daphne::MatrixRepresentation " +
                std::to_string(static_cast<int>(rep)));
    }
}

mlir::daphne::MatrixRepresentation mlir::daphne::stringToMatrixRepresentation(const std::string &str) {
    if(str == "dense")
        return MatrixRepresentation::Dense;
    else if (str == "sparse")
        return MatrixRepresentation::Sparse;
    else
        throw std::runtime_error("No matrix representation equals the string `" + str + "`");
}

namespace mlir::daphne {
    namespace detail {
        struct MatrixTypeStorage : public ::mlir::TypeStorage {
            // TODO: adapt epsilon for equality check (I think the only use is saving memory for the MLIR-IR representation of this type)
            //  the choosen epsilon directly defines how accurate our sparsity inference can be
            constexpr static const double epsilon = 1e-6;
            MatrixTypeStorage(::mlir::Type elementType,
                              ssize_t numRows,
                              ssize_t numCols,
                              double sparsity,
                              MatrixRepresentation representation)
                : elementType(elementType), numRows(numRows), numCols(numCols), sparsity(sparsity),
                  representation(representation) {}

            /// The hash key is a tuple of the parameter types.
            using KeyTy = std::tuple<::mlir::Type, ssize_t, ssize_t, double, MatrixRepresentation>;
            bool operator==(const KeyTy &tblgenKey) const {
                if(!(elementType == std::get<0>(tblgenKey)))
                    return false;
                if(numRows != std::get<1>(tblgenKey))
                    return false;
                if(numCols != std::get<2>(tblgenKey))
                    return false;
                if(std::fabs(sparsity - std::get<3>(tblgenKey)) >= epsilon)
                    return false;
                if(representation != std::get<4>(tblgenKey))
                    return false;
                return true;
            }
            static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
                auto float_hashable = static_cast<ssize_t>(std::get<3>(tblgenKey) / epsilon);
                return ::llvm::hash_combine(std::get<0>(tblgenKey),
                    std::get<1>(tblgenKey),
                    std::get<2>(tblgenKey),
                    float_hashable,
                    std::get<4>(tblgenKey));
            }

            /// Define a construction method for creating a new instance of this
            /// storage.
            static MatrixTypeStorage *construct(::mlir::TypeStorageAllocator &allocator,
                                                const KeyTy &tblgenKey) {
                auto elementType = std::get<0>(tblgenKey);
                auto numRows = std::get<1>(tblgenKey);
                auto numCols = std::get<2>(tblgenKey);
                auto sparsity = std::get<3>(tblgenKey);
                auto representation = std::get<4>(tblgenKey);

                return new(allocator.allocate<MatrixTypeStorage>())
                    MatrixTypeStorage(elementType, numRows, numCols, sparsity, representation);
            }
            ::mlir::Type elementType;
            ssize_t numRows;
            ssize_t numCols;
            double sparsity;
            MatrixRepresentation representation;
        };
    }
    ::mlir::Type MatrixType::getElementType() const { return getImpl()->elementType; }
    ssize_t MatrixType::getNumRows() const { return getImpl()->numRows; }
    ssize_t MatrixType::getNumCols() const { return getImpl()->numCols; }
    double MatrixType::getSparsity() const { return getImpl()->sparsity; }
    MatrixRepresentation MatrixType::getRepresentation() const { return getImpl()->representation; }
}

mlir::OpFoldResult mlir::daphne::ConstantOp::fold(mlir::ArrayRef<mlir::Attribute> operands)
{
    assert(operands.empty() && "constant has no operands");
    return value();
}

::mlir::LogicalResult mlir::daphne::MatrixType::verify(
        ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
        Type elementType,
        ssize_t numRows, ssize_t numCols, double sparsity, MatrixRepresentation rep
)
{
    if (
        (
            // Value type is unknown.
            elementType.isa<mlir::daphne::UnknownType>()
            // Value type is known.
            || elementType.isSignedInteger(64)
            || elementType.isUnsignedInteger(8)
            || elementType.isF32()
            || elementType.isF64()
            || elementType.isIndex()
        ) && (
            // Number of rows and columns are valid (-1 for unknown).
            numRows >= -1 && numCols >= -1
        ) && (
            sparsity == -1 || (sparsity >= 0.0 && sparsity <= 1.0)
        )
    )
        return mlir::success();
    else
        return emitError() << "invalid matrix element type: " << elementType;
}

::mlir::LogicalResult mlir::daphne::FrameType::verify(
        ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
        std::vector<Type> columnTypes,
        ssize_t numRows, ssize_t numCols,
        std::vector<std::string> * labels
)
{
    // TODO Verify the individual column types.
    if(numRows < -1 || numCols < -1)
        return mlir::failure();
    if(numCols != -1) {
        if(static_cast<ssize_t>(columnTypes.size()) != numCols)
            return mlir::failure();
        if(labels && static_cast<ssize_t>(labels->size()) != numCols)
            return mlir::failure();
    }
    if(labels && labels->size() != columnTypes.size())
        return mlir::failure();
    return mlir::success();
}

::mlir::LogicalResult mlir::daphne::HandleType::verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
                                                       Type dataType)
{
    if (dataType.isa<MatrixType>()) {
        return mlir::success();
    }
    else
        return emitError() << "only matrix type is supported for handle atm, got: " << dataType;
}

mlir::LogicalResult mlir::daphne::VectorizedPipelineOp::canonicalize(mlir::daphne::VectorizedPipelineOp op,
                                                                     mlir::PatternRewriter &rewriter)
{
    std::vector<Value> resultsToReplace;
    std::vector<Value> outRows;
    std::vector<Value> outCols;
    std::vector<Attribute> vCombineAttrs;

    llvm::BitVector eraseIxs;
    eraseIxs.resize(op.getNumResults());
    for(auto result : op.getResults()) {
        auto resultIx = result.getResultNumber();
        if(result.use_empty()) {
            // remove
            eraseIxs.set(resultIx);
        }
        else {
            resultsToReplace.push_back(result);
            outRows.push_back(op.out_rows()[resultIx]);
            outCols.push_back(op.out_cols()[resultIx]);
            vCombineAttrs.push_back(op.combines()[resultIx]);
        }
    }
    op.body().front().getTerminator()->eraseOperands(eraseIxs);
    if(!op.cuda().getBlocks().empty())
        op.cuda().front().getTerminator()->eraseOperands(eraseIxs);

    if(resultsToReplace.size() == op->getNumResults()) {
        return failure();
    }
    auto pipelineOp = rewriter.create<daphne::VectorizedPipelineOp>(op.getLoc(),
        ValueRange(resultsToReplace).getTypes(),
        op.inputs(),
        outRows,
        outCols,
        op.splits(),
        rewriter.getArrayAttr(vCombineAttrs),
        op.ctx());
    pipelineOp.body().takeBody(op.body());
    if(!op.cuda().getBlocks().empty())
        pipelineOp.cuda().takeBody(op.cuda());
    for (auto e : llvm::enumerate(resultsToReplace)) {
        auto resultToReplace = e.value();
        auto i = e.index();
        resultToReplace.replaceAllUsesWith(pipelineOp.getResult(i));
    }
    op.erase();
    return success();
}

// ****************************************************************************
// Fold utility functions/macros
// ****************************************************************************
// For families of operations.

// Adapted from "mlir/Dialect/CommonFolders.h"
template<class AttrElementT,
    class ElementValueT = typename AttrElementT::ValueType,
    class CalculationT = std::function<ElementValueT(const ElementValueT &, const ElementValueT &)>>
mlir::Attribute constFoldBinaryOp(mlir::Type resultType, llvm::ArrayRef<mlir::Attribute> operands,
                                  const CalculationT &calculate) {
    assert(operands.size() == 2 && "binary op takes two operands");
    if(!operands[0] || !operands[1])
        return {};
    if(operands[0].getType() != operands[1].getType())
        return {};

    if(operands[0].isa<AttrElementT>() && operands[1].isa<AttrElementT>()) {
        auto lhs = operands[0].cast<AttrElementT>();
        auto rhs = operands[1].cast<AttrElementT>();

        return AttrElementT::get(resultType, calculate(lhs.getValue(), rhs.getValue()));
    }
    return {};
}
template<class AttrElementT,
    class ElementValueT = typename AttrElementT::ValueType,
    class CalculationT = std::function<bool(const ElementValueT &, const ElementValueT &)>>
mlir::Attribute constFoldBinaryCmpOp(llvm::ArrayRef<mlir::Attribute> operands,
                                     const CalculationT &calculate) {
    assert(operands.size() == 2 && "binary op takes two operands");
    if(!operands[0] || !operands[1])
        return {};
    if(operands[0].getType() != operands[1].getType())
        return {};

    if(operands[0].isa<AttrElementT>() && operands[1].isa<AttrElementT>()) {
        auto lhs = operands[0].cast<AttrElementT>();
        auto rhs = operands[1].cast<AttrElementT>();
        return mlir::BoolAttr::get(lhs.getContext(), calculate(lhs.getValue(), rhs.getValue()));
    }
    return {};
}

// ****************************************************************************
// Fold implementations
// ****************************************************************************

mlir::OpFoldResult mlir::daphne::CastOp::fold(ArrayRef<Attribute> operands) {
    if (isTrivialCast()) {
        if (operands[0])
            return {operands[0]};
        else
            return {arg()};
    }
    if(auto in = operands[0].dyn_cast_or_null<IntegerAttr>()) {
        auto apInt = in.getValue();
        if(auto outTy = getType().dyn_cast<IntegerType>()) {
            // TODO: throw exception if bits truncated?
            if(outTy.isUnsignedInteger()) {
                apInt = apInt.zextOrTrunc(outTy.getWidth());
            }
            else if(outTy.isSignedInteger()) {
                apInt = apInt.sextOrTrunc(outTy.getWidth());
            }
            return IntegerAttr::getChecked(getLoc(), outTy, apInt);
        }
        if(auto outTy = getType().dyn_cast<IndexType>()) {
            return IntegerAttr::getChecked(getLoc(), outTy, apInt);
        }
        if(getType().isF64()) {
            if(in.getType().isSignedInteger()) {
                return FloatAttr::getChecked(getLoc(),
                    getType(),
                    llvm::APIntOps::RoundSignedAPIntToDouble(in.getValue()));
            }
            if(in.getType().isUnsignedInteger() || in.getType().isIndex()) {
                return FloatAttr::getChecked(getLoc(), getType(), llvm::APIntOps::RoundAPIntToDouble(in.getValue()));
            }
        }
        if(getType().isF32()) {
            if(in.getType().isSignedInteger()) {
                return FloatAttr::getChecked(getLoc(),
                    getType(),
                    llvm::APIntOps::RoundSignedAPIntToFloat(in.getValue()));
            }
            if(in.getType().isUnsignedInteger()) {
                return FloatAttr::get(getType(), llvm::APIntOps::RoundAPIntToFloat(in.getValue()));
            }
        }
    }
    if(auto in = operands[0].dyn_cast_or_null<FloatAttr>()) {
        auto val = in.getValueAsDouble();
        if(getType().isF64()) {
            return FloatAttr::getChecked(getLoc(), getType(), val);
        }
        if(getType().isF32()) {
            return FloatAttr::getChecked(getLoc(), getType(), static_cast<float>(val));
        }
        if(getType().isIntOrIndex()) {
            auto num = static_cast<int64_t>(val);
            return IntegerAttr::getChecked(getLoc(), getType(), num);
        }
    }
    return {};
}

mlir::OpFoldResult mlir::daphne::EwAddOp::fold(ArrayRef<Attribute> operands) {
    auto floatOp = [](const llvm::APFloat &a, const llvm::APFloat &b) { return a + b; };
    // TODO: we could check overflows
    auto intOp = [](const llvm::APInt &a, const llvm::APInt &b) { return a + b; };
    if(auto res = constFoldBinaryOp<FloatAttr>(getType(), operands, floatOp))
        return res;
    if(auto res = constFoldBinaryOp<IntegerAttr>(getType(), operands, intOp))
        return res;
    return {};
}

mlir::OpFoldResult mlir::daphne::EwSubOp::fold(ArrayRef<Attribute> operands) {
    auto floatOp = [](const llvm::APFloat &a, const llvm::APFloat &b) { return a - b; };
    auto intOp = [](const llvm::APInt &a, const llvm::APInt &b) { return a - b; };
    if(auto res = constFoldBinaryOp<FloatAttr>(getType(), operands, floatOp))
        return res;
    if(auto res = constFoldBinaryOp<IntegerAttr>(getType(), operands, intOp))
        return res;
    return {};
}

mlir::OpFoldResult mlir::daphne::EwMulOp::fold(ArrayRef<Attribute> operands) {
    auto floatOp = [](const llvm::APFloat &a, const llvm::APFloat &b) { return a * b; };
    auto intOp = [](const llvm::APInt &a, const llvm::APInt &b) { return a * b; };
    if(auto res = constFoldBinaryOp<FloatAttr>(getType(), operands, floatOp))
        return res;
    if(auto res = constFoldBinaryOp<IntegerAttr>(getType(), operands, intOp))
        return res;
    return {};
}

mlir::OpFoldResult mlir::daphne::EwDivOp::fold(ArrayRef<Attribute> operands) {
    auto floatOp = [](const llvm::APFloat &a, const llvm::APFloat &b) { return a / b; };
    auto sintOp = [&](const llvm::APInt &a, const llvm::APInt &b) {
        if(b == 0) {
            std::string msg = "Can't divide by 0";
            emitError() << msg;
            throw std::runtime_error(msg);
        }
        return a.sdiv(b);
    };
    auto uintOp = [&](const llvm::APInt &a, const llvm::APInt &b) {
        if(b == 0) {
            std::string msg = "Can't divide by 0";
            emitError() << msg;
            throw std::runtime_error(msg);
        }
        return a.udiv(b);
    };

    if(auto res = constFoldBinaryOp<FloatAttr>(getType(), operands, floatOp))
        return res;
    if(getType().isSignedInteger()) {
        if(auto res = constFoldBinaryOp<IntegerAttr>(getType(), operands, sintOp))
            return res;
    }
    else if(getType().isUnsignedInteger()) {
        if(auto res = constFoldBinaryOp<IntegerAttr>(getType(), operands, uintOp))
            return res;
    }
    return {};
}

mlir::OpFoldResult mlir::daphne::EwPowOp::fold(ArrayRef<Attribute> operands) {
    // TODO: EwPowOp integer constant folding
    auto floatOp = [](const llvm::APFloat &a, const llvm::APFloat &b) {
        return std::pow(a.convertToDouble(), b.convertToDouble());
    };
    if(auto res = constFoldBinaryOp<FloatAttr>(getType(), operands, floatOp))
        return res;
    return {};
}

mlir::OpFoldResult mlir::daphne::EwModOp::fold(ArrayRef<Attribute> operands) {
    auto sintOp = [&](const llvm::APInt &a, const llvm::APInt &b) {
        if(b == 0) {
            std::string msg = "Can't compute mod 0";
            emitError() << msg;
            throw std::runtime_error(msg);
        }
        return a.srem(b);
    };
    auto uintOp = [&](const llvm::APInt &a, const llvm::APInt &b) {
        if(b == 0) {
            std::string msg = "Can't compute mod 0";
            emitError() << msg;
            throw std::runtime_error(msg);
        }
        return a.urem(b);
    };
    if(getType().isSignedInteger()) {
        if(auto res = constFoldBinaryOp<IntegerAttr>(getType(), operands, sintOp))
            return res;
    }
    else if(getType().isUnsignedInteger()) {
        if(auto res = constFoldBinaryOp<IntegerAttr>(getType(), operands, uintOp))
            return res;
    }
    return {};
}

mlir::OpFoldResult mlir::daphne::EwLogOp::fold(ArrayRef<Attribute> operands) {
    auto floatOp = [](const llvm::APFloat &a, const llvm::APFloat &b) {
        // Equivalent to log_b(a)
        return ilogb(a) / ilogb(b);
    };
    if(auto res = constFoldBinaryOp<FloatAttr>(getType(), operands, floatOp))
        return res;
    return {};
}

mlir::OpFoldResult mlir::daphne::EwMinOp::fold(ArrayRef<Attribute> operands) {
    auto floatOp = [](const llvm::APFloat &a, const llvm::APFloat &b) { return llvm::minimum(a, b); };
    auto sintOp = [&](const llvm::APInt &a, const llvm::APInt &b) {
        if(a.slt(b))
            return a;
        else
            return b;
    };
    auto uintOp = [&](const llvm::APInt &a, const llvm::APInt &b) {
        if(a.ult(b))
            return a;
        else
            return b;
    };
    if(auto res = constFoldBinaryOp<FloatAttr>(getType(), operands, floatOp))
        return res;
    if(getType().isSignedInteger()) {
        if(auto res = constFoldBinaryOp<IntegerAttr>(getType(), operands, sintOp))
            return res;
    }
    else if(getType().isUnsignedInteger()) {
        if(auto res = constFoldBinaryOp<IntegerAttr>(getType(), operands, uintOp))
            return res;
    }
    return {};
}

mlir::OpFoldResult mlir::daphne::EwMaxOp::fold(ArrayRef<Attribute> operands) {
    auto floatOp = [](const llvm::APFloat &a, const llvm::APFloat &b) { return llvm::maximum(a, b); };
    auto sintOp = [&](const llvm::APInt &a, const llvm::APInt &b) {
        if(a.sgt(b))
            return a;
        else
            return b;
    };
    auto uintOp = [&](const llvm::APInt &a, const llvm::APInt &b) {
        if(a.ugt(b))
            return a;
        else
            return b;
    };
    if(auto res = constFoldBinaryOp<FloatAttr>(getType(), operands, floatOp))
        return res;
    if(getType().isSignedInteger()) {
        if(auto res = constFoldBinaryOp<IntegerAttr>(getType(), operands, sintOp))
            return res;
    }
    else if(getType().isUnsignedInteger()) {
        if(auto res = constFoldBinaryOp<IntegerAttr>(getType(), operands, uintOp))
            return res;
    }
    return {};
}

mlir::OpFoldResult mlir::daphne::EwAndOp::fold(ArrayRef<Attribute> operands) {
    auto boolOp = [](const bool &a, const bool &b) { return a && b; };
    auto intOp = [](const llvm::APInt &a, const llvm::APInt &b) { return (a != 0) && (b != 0); };
    if(auto res = constFoldBinaryCmpOp<BoolAttr>(operands, boolOp))
        return res;
    // TODO: should output bool?
    if(auto res = constFoldBinaryOp<IntegerAttr>(getType(), operands, intOp))
        return res;
    return {};
}

mlir::OpFoldResult mlir::daphne::EwOrOp::fold(ArrayRef<Attribute> operands) {
    auto boolOp = [](const bool &a, const bool &b) { return a || b; };
    auto intOp = [](const llvm::APInt &a, const llvm::APInt &b) { return (a != 0) || (b != 0); };
    if(auto res = constFoldBinaryCmpOp<BoolAttr>(operands, boolOp))
        return res;
    // TODO: should output bool
    if(auto res = constFoldBinaryOp<IntegerAttr>(getType(), operands, intOp))
        return res;
    return {};
}

mlir::OpFoldResult mlir::daphne::EwXorOp::fold(ArrayRef<Attribute> operands) {
    auto boolOp = [](const bool &a, const bool &b) { return a ^ b; };
    auto intOp = [](const llvm::APInt &a, const llvm::APInt &b) { return (a != 0) ^ (b != 0); };
    if(auto res = constFoldBinaryCmpOp<BoolAttr>(operands, boolOp))
        return res;
    // TODO: should output bool
    if(auto res = constFoldBinaryOp<IntegerAttr>(getType(), operands, intOp))
        return res;
    return {};
}

mlir::OpFoldResult mlir::daphne::EwConcatOp::fold(ArrayRef<Attribute> operands) {
    assert(operands.size() == 2 && "binary op takes two operands");
    if(!operands[0] || !operands[1])
        return {};
    if(operands[0].getType() != operands[1].getType())
        return {};

    if(operands[0].isa<StringAttr>() && operands[1].isa<StringAttr>()) {
        auto lhs = operands[0].cast<StringAttr>();
        auto rhs = operands[1].cast<StringAttr>();

        auto concated = lhs.getValue().str() + rhs.getValue().str();
        return StringAttr::get(concated, getType());
    }
    return {};
}

// TODO This is duplicated from EwConcatOp. Actually, ConcatOp itself is only
// a temporary workaround, so it should be removed altogether later.
mlir::OpFoldResult mlir::daphne::ConcatOp::fold(ArrayRef<Attribute> operands) {
    assert(operands.size() == 2 && "binary op takes two operands");
    if(!operands[0] || !operands[1])
        return {};
    if(operands[0].getType() != operands[1].getType())
        return {};

    if(operands[0].isa<StringAttr>() && operands[1].isa<StringAttr>()) {
        auto lhs = operands[0].cast<StringAttr>();
        auto rhs = operands[1].cast<StringAttr>();

        auto concated = lhs.getValue().str() + rhs.getValue().str();
        return StringAttr::get(concated, getType());
    }
    return {};
}

mlir::OpFoldResult mlir::daphne::EwEqOp::fold(ArrayRef<Attribute> operands) {
    auto floatOp = [](const llvm::APFloat &a, const llvm::APFloat &b) { return a == b; };
    auto intOp = [](const llvm::APInt &a, const llvm::APInt &b) { return a == b; };
    // TODO: fix bool return
    if(auto res = constFoldBinaryOp<FloatAttr>(getType(), operands, floatOp))
        return res;
    if(auto res = constFoldBinaryOp<IntegerAttr>(getType(), operands, intOp))
        return res;
    return {};
}

mlir::OpFoldResult mlir::daphne::EwNeqOp::fold(ArrayRef<Attribute> operands) {
    auto floatOp = [](const llvm::APFloat &a, const llvm::APFloat &b) { return a != b; };
    auto intOp = [](const llvm::APInt &a, const llvm::APInt &b) { return a != b; };
    // TODO: fix bool return
    if(auto res = constFoldBinaryOp<FloatAttr>(getType(), operands, floatOp))
        return res;
    if(auto res = constFoldBinaryOp<IntegerAttr>(getType(), operands, intOp))
        return res;
    return {};
}

mlir::OpFoldResult mlir::daphne::EwLtOp::fold(ArrayRef<Attribute> operands) {
    auto floatOp = [](const llvm::APFloat &a, const llvm::APFloat &b) { return a < b; };
    auto sintOp = [](const llvm::APInt &a, const llvm::APInt &b) { return a.slt(b); };
    auto uintOp = [](const llvm::APInt &a, const llvm::APInt &b) { return a.ult(b); };
    // TODO: fix bool return
    if(auto res = constFoldBinaryOp<FloatAttr>(getType(), operands, floatOp))
        return res;
    if(getType().isSignedInteger()) {
        if(auto res = constFoldBinaryOp<IntegerAttr>(getType(), operands, sintOp))
            return res;
    }
    else if(getType().isUnsignedInteger()) {
        if(auto res = constFoldBinaryOp<IntegerAttr>(getType(), operands, uintOp))
            return res;
    }
    return {};
}

mlir::OpFoldResult mlir::daphne::EwLeOp::fold(ArrayRef<Attribute> operands) {
    auto floatOp = [](const llvm::APFloat &a, const llvm::APFloat &b) { return a <= b; };
    auto sintOp = [](const llvm::APInt &a, const llvm::APInt &b) { return a.sle(b); };
    auto uintOp = [](const llvm::APInt &a, const llvm::APInt &b) { return a.ule(b); };
    // TODO: fix bool return
    if(auto res = constFoldBinaryOp<FloatAttr>(getType(), operands, floatOp))
        return res;
    if(getType().isSignedInteger()) {
        if(auto res = constFoldBinaryOp<IntegerAttr>(getType(), operands, sintOp))
            return res;
    }
    else if(getType().isUnsignedInteger()) {
        if(auto res = constFoldBinaryOp<IntegerAttr>(getType(), operands, uintOp))
            return res;
    }
    return {};
}

mlir::OpFoldResult mlir::daphne::EwGtOp::fold(ArrayRef<Attribute> operands) {
    auto floatOp = [](const llvm::APFloat &a, const llvm::APFloat &b) { return a > b; };
    auto sintOp = [](const llvm::APInt &a, const llvm::APInt &b) { return a.sgt(b); };
    auto uintOp = [](const llvm::APInt &a, const llvm::APInt &b) { return a.ugt(b); };
    // TODO: fix bool return
    if(auto res = constFoldBinaryOp<FloatAttr>(getType(), operands, floatOp))
        return res;
    if(getType().isSignedInteger()) {
        if(auto res = constFoldBinaryOp<IntegerAttr>(getType(), operands, sintOp))
            return res;
    }
    else if(getType().isUnsignedInteger()) {
        if(auto res = constFoldBinaryOp<IntegerAttr>(getType(), operands, uintOp))
            return res;
    }
    return {};
}

mlir::OpFoldResult mlir::daphne::EwGeOp::fold(ArrayRef<Attribute> operands) {
    auto floatOp = [](const llvm::APFloat &a, const llvm::APFloat &b) { return a >= b; };
    auto sintOp = [](const llvm::APInt &a, const llvm::APInt &b) { return a.sge(b); };
    auto uintOp = [](const llvm::APInt &a, const llvm::APInt &b) { return a.uge(b); };
    // TODO: fix bool return
    if(auto res = constFoldBinaryOp<FloatAttr>(getType(), operands, floatOp))
        return res;
    if(getType().isSignedInteger()) {
        if(auto res = constFoldBinaryOp<IntegerAttr>(getType(), operands, sintOp))
            return res;
    }
    else if(getType().isUnsignedInteger()) {
        if(auto res = constFoldBinaryOp<IntegerAttr>(getType(), operands, uintOp))
            return res;
    }
    return {};
}

/**
 * @brief Replaces NumRowsOp by a constant, if the #rows of the input is known
 * (e.g., due to shape inference).
 */
mlir::LogicalResult mlir::daphne::NumRowsOp::canonicalize(
        mlir::daphne::NumRowsOp op, PatternRewriter &rewriter
) {
    ssize_t numRows = -1;
    
    mlir::Type inTy = op.arg().getType();
    if(auto t = inTy.dyn_cast<mlir::daphne::MatrixType>())
        numRows = t.getNumRows();
    else if(auto t = inTy.dyn_cast<mlir::daphne::FrameType>())
        numRows = t.getNumRows();
    
    if(numRows != -1) {
        rewriter.replaceOpWithNewOp<mlir::daphne::ConstantOp>(
                op, rewriter.getIndexAttr(numRows)
        );
        return mlir::success();
    }
    return mlir::failure();
}

/**
 * @brief Replaces NumColsOp by a constant, if the #cols of the input is known
 * (e.g., due to shape inference).
 */
mlir::LogicalResult mlir::daphne::NumColsOp::canonicalize(
        mlir::daphne::NumColsOp op, PatternRewriter &rewriter
) {
    ssize_t numCols = -1;
    
    mlir::Type inTy = op.arg().getType();
    if(auto t = inTy.dyn_cast<mlir::daphne::MatrixType>())
        numCols = t.getNumCols();
    else if(auto t = inTy.dyn_cast<mlir::daphne::FrameType>())
        numCols = t.getNumCols();
    
    if(numCols != -1) {
        rewriter.replaceOpWithNewOp<mlir::daphne::ConstantOp>(
                op, rewriter.getIndexAttr(numCols)
        );
        return mlir::success();
    }
    return mlir::failure();
}

/**
 * @brief Replaces NumCellsOp by a constant, if the #rows and #cols of the
 * input is known (e.g., due to shape inference).
 */
mlir::LogicalResult mlir::daphne::NumCellsOp::canonicalize(
        mlir::daphne::NumCellsOp op, PatternRewriter &rewriter
) {
    ssize_t numRows = -1;
    ssize_t numCols = -1;
    
    mlir::Type inTy = op.arg().getType();
    if(auto t = inTy.dyn_cast<mlir::daphne::MatrixType>()) {
        numRows = t.getNumRows();
        numCols = t.getNumCols();
    }
    else if(auto t = inTy.dyn_cast<mlir::daphne::FrameType>()) {
        numRows = t.getNumRows();
        numCols = t.getNumCols();
    }
    
    if(numRows != -1 && numCols != -1) {
        rewriter.replaceOpWithNewOp<mlir::daphne::ConstantOp>(
                op, rewriter.getIndexAttr(numRows * numCols)
        );
        return mlir::success();
    }
    return mlir::failure();
}

/**
 * @brief Replaces a `DistributeOp` by a `DistributedReadOp`, if its input
 * value (a) is defined by a `ReadOp`, and (b) is not used elsewhere.
 * @param context
 */
struct SimplifyDistributeRead : public mlir::OpRewritePattern<mlir::daphne::DistributeOp> {
    SimplifyDistributeRead(mlir::MLIRContext *context)
        : OpRewritePattern<mlir::daphne::DistributeOp>(context, 1) {
        //
    }
    
    mlir::LogicalResult
    matchAndRewrite(
            mlir::daphne::DistributeOp op, mlir::PatternRewriter &rewriter
    ) const override {
        mlir::daphne::ReadOp readOp = op.mat().getDefiningOp<mlir::daphne::ReadOp>();
        if(!readOp || !readOp.getOperation()->hasOneUse())
            return mlir::failure();
        rewriter.replaceOp(
                op, {rewriter.create<mlir::daphne::DistributedReadOp>(
                        readOp.getLoc(), op.getType(), readOp.fileName()
                )}
        );
        // TODO Instead of erasing the ReadOp here, the compiler should
        // generally remove unused SSA values. Then, we might even drop the
        // hasOneUse requirement above.
        rewriter.eraseOp(readOp);
        return mlir::success();
    }
};

/**
 * @brief Turns an addition into a concatenation, if one of the inputs is a
 * string.
 * 
 * This is important, since we use the '+'-operator for both addition and
 * string concatenation in DaphneDSL, while the types of the operands might be
 * known only after type inference.
 * 
 * @param op
 * @param rewriter
 * @return 
 */
mlir::LogicalResult mlir::daphne::EwAddOp::canonicalize(
        mlir::daphne::EwAddOp op, PatternRewriter &rewriter
) {
    mlir::Value lhs = op.lhs();
    mlir::Value rhs = op.rhs();
    const bool lhsIsStr = lhs.getType().isa<mlir::daphne::StringType>();
    const bool rhsIsStr = rhs.getType().isa<mlir::daphne::StringType>();
    if(lhsIsStr || rhsIsStr) {
        mlir::Type strTy = mlir::daphne::StringType::get(rewriter.getContext());
        if(!lhsIsStr)
            lhs = rewriter.create<mlir::daphne::CastOp>(op.getLoc(), strTy, lhs);
        if(!rhsIsStr)
            rhs = rewriter.create<mlir::daphne::CastOp>(op.getLoc(), strTy, rhs);
        rewriter.replaceOpWithNewOp<mlir::daphne::ConcatOp>(op, strTy, lhs, rhs);
        return mlir::success();
    }
    return mlir::failure();
}

void mlir::daphne::DistributeOp::getCanonicalizationPatterns(
        RewritePatternSet &results, MLIRContext *context
) {
    results.add<SimplifyDistributeRead>(context);
}