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

#include <compiler/utils/CompilerUtils.h>
#include <util/ErrorHandler.h>
#include <ir/daphneir/Daphne.h>

#include <ir/daphneir/DaphneOpsEnums.cpp.inc>

#include "mlir/Support/LogicalResult.h"
#define GET_OP_CLASSES
#include <ir/daphneir/DaphneOps.cpp.inc>
#define GET_TYPEDEF_CLASSES
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/APSInt.h>
#include <llvm/ADT/BitVector.h>

#include <ir/daphneir/DaphneOpsDialect.cpp.inc>
#include <ir/daphneir/DaphneOpsTypes.cpp.inc>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Transforms/InliningUtils.h"

#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/APSInt.h>
#include <llvm/ADT/DenseMap.h>

struct DaphneInlinerInterface : public mlir::DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(mlir::Operation *call, mlir::Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  bool isLegalToInline(mlir::Operation *, mlir::Region *, bool, mlir::IRMapping &) const final {
    return true;
  }

  bool isLegalToInline(mlir::Region *, mlir::Region *, bool, mlir::IRMapping &) const final {
    return true;
  }

  void handleTerminator(mlir::Operation *op,
                        mlir::ArrayRef<mlir::Value> valuesToRepl) const final {
    auto returnOp = mlir::dyn_cast<mlir::daphne::ReturnOp>(op);

    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }

  mlir::Operation *materializeCallConversion(mlir::OpBuilder &builder, mlir::Value input,
                                       mlir::Type resultType,
                                       mlir::Location conversionLoc) const final {
    return builder.create<mlir::daphne::CastOp>(conversionLoc, resultType, input);
  }
};

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
    addInterfaces<DaphneInlinerInterface>();
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
    mlir::ParseResult pr = parser.parseKeyword(&keyword);
    if(mlir::failed(pr))
        throw std::runtime_error("parsing a DaphneIR type failed");
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
        if (parser.parseLess()) {
            return nullptr;
        }
        if (parser.parseOptionalQuestion()) {
            // Parse #rows if there was no '?'.
            if (parser.parseInteger<ssize_t>(numRows)) {
                return nullptr;
            }
        }
        if (parser.parseXInDimensionList()) {
            return nullptr;
        }
        if (parser.parseOptionalQuestion()) {
            // Parse #cols if there was no '?'.
            if (parser.parseInteger<ssize_t>(numCols)) {
                return nullptr;
            }
        }
        if (parser.parseXInDimensionList() ||
            parser.parseType(elementType)
        ) {
            return nullptr;
        }
        // additional properties (only print/read them when present, as this will probably get more and more)
        while (succeeded(parser.parseOptionalColon())) {
            if (succeeded(parser.parseOptionalKeyword("sp"))) {
                if (sparsity != -1.0) {
                    // read sparsity twice
                    return nullptr;
                }
                if (parser.parseLSquare() || parser.parseFloat(sparsity) || parser.parseRSquare()) {
                    return nullptr;
                }
            }
            else if (succeeded(parser.parseOptionalKeyword("rep"))) {
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
    else if (keyword == "DaphneContext") {
        return mlir::daphne::DaphneContextType::get(parser.getBuilder().getContext());
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
    if (type.isa<mlir::daphne::StructureType>())
        os << "Structure";
    else if (auto t = type.dyn_cast<mlir::daphne::MatrixType>()) {
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
    else if (isa<mlir::daphne::StringType>(type))
        os << "String";
    else if (auto t = type.dyn_cast<mlir::daphne::VariadicPackType>())
        os << "VariadicPack<" << t.getContainedType() << '>';
    else if (isa<mlir::daphne::DaphneContextType>(type))
        os << "DaphneContext";
    else if (isa<mlir::daphne::FileType>(type))
        os << "File";
    else if (isa<mlir::daphne::DescriptorType>(type))
        os << "Descriptor";
    else if (isa<mlir::daphne::TargetType>(type))
        os << "Target";
    else if (isa<mlir::daphne::UnknownType>(type))
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

mlir::OpFoldResult mlir::daphne::ConstantOp::fold(FoldAdaptor adaptor)
{
    assert(adaptor.getOperands().empty() && "constant has no operands");
    return getValue();
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
            llvm::isa<mlir::daphne::UnknownType>(elementType)
            // Value type is known.
            || elementType.isSignedInteger(64)
            || elementType.isUnsignedInteger(8)
            || elementType.isUnsignedInteger(64)
            || elementType.isF32()
            || elementType.isF64()
            || elementType.isIndex()
            || elementType.isInteger(1)
            || llvm::isa<mlir::daphne::StringType>(elementType)
            || elementType.isUnsignedInteger(64)
            || elementType.isUnsignedInteger(32)
            || elementType.isSignedInteger(32)
            || elementType.isSignedInteger(8)
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
        // ToDo: ExtractColOp does not provide these columnTypes
        if(!columnTypes.empty()) {
            if (static_cast<ssize_t>(columnTypes.size()) != numCols)
                return mlir::failure();
            if (labels && static_cast<ssize_t>(labels->size()) != numCols)
                return mlir::failure();
        }
    }
    if(labels && labels->size() != columnTypes.size())
        return mlir::failure();
    return mlir::success();
}

::mlir::LogicalResult mlir::daphne::HandleType::verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
                                                       Type dataType)
{
    if (llvm::isa<MatrixType>(dataType)) {
        return mlir::success();
    }
    else
        return emitError() << "only matrix type is supported for handle atm, got: " << dataType;
}

mlir::LogicalResult mlir::daphne::VectorizedPipelineOp::canonicalize(mlir::daphne::VectorizedPipelineOp op,
                                                                     mlir::PatternRewriter &rewriter)
{
    // // Find duplicate inputs
    std::vector<Attribute> vSplitsAttrs;
    for (auto & split : op.getSplits())
        vSplitsAttrs.push_back(split);
    auto currentSize = op.getInputs().size();
    
    DenseMap<Value, size_t> inputMap;

    for (size_t i = 0; i < currentSize; i++) {
        const auto& input = op.getInputs()[i];
        const auto& split = op.getSplits()[i].cast<daphne::VectorSplitAttr>().getValue();

        if (inputMap.count(input) == 0) {
            inputMap[input] = i;
        } else {
            size_t j = inputMap[input];
            if (op.getSplits()[j].cast<daphne::VectorSplitAttr>().getValue() == split) {
                op.getBody().getArgument(i).replaceAllUsesWith(op.getBody().getArgument(j));
                op.getBody().eraseArgument(i);
                op.getInputsMutable().erase(i);
                vSplitsAttrs.erase(vSplitsAttrs.begin() + i);
                currentSize--;
                i--;
            }
        }
    }

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
            outRows.push_back(op.getOutRows()[resultIx]);
            outCols.push_back(op.getOutCols()[resultIx]);
            vCombineAttrs.push_back(op.getCombines()[resultIx]);
        }
    }
    op.getBody().front().getTerminator()->eraseOperands(eraseIxs);
    if(!op.getCuda().getBlocks().empty())
        op.getCuda().front().getTerminator()->eraseOperands(eraseIxs);

    if(resultsToReplace.size() == op->getNumResults() && op.getSplits().size() == vSplitsAttrs.size()) {
        return failure();
    }
    auto pipelineOp = rewriter.create<daphne::VectorizedPipelineOp>(op.getLoc(),
        ValueRange(resultsToReplace).getTypes(),
        op.getInputs(),
        outRows,
        outCols,
        rewriter.getArrayAttr(vSplitsAttrs),
        rewriter.getArrayAttr(vCombineAttrs),
        op.getCtx());
    pipelineOp.getBody().takeBody(op.getBody());
    if(!op.getCuda().getBlocks().empty())
        pipelineOp.getCuda().takeBody(op.getCuda());
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

    if(llvm::isa<AttrElementT>(operands[0]) && llvm::isa<AttrElementT>(operands[1])) {
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

    if(llvm::isa<AttrElementT>(operands[0]) && llvm::isa<AttrElementT>(operands[1])) {
        auto lhs = operands[0].cast<AttrElementT>();
        auto rhs = operands[1].cast<AttrElementT>();
        return mlir::BoolAttr::get(lhs.getContext(), calculate(lhs.getValue(), rhs.getValue()));
    }
    return {};
}

// ****************************************************************************
// Fold implementations
// ****************************************************************************

mlir::OpFoldResult mlir::daphne::CastOp::fold(FoldAdaptor adaptor) {
    ArrayRef<Attribute> operands = adaptor.getOperands();
    if (isTrivialCast()) {
        if (operands[0])
            return {operands[0]};
        else
            return {getArg()};
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

mlir::OpFoldResult mlir::daphne::EwAddOp::fold(FoldAdaptor adaptor) {
    ArrayRef<Attribute> operands = adaptor.getOperands();
    auto floatOp = [](const llvm::APFloat &a, const llvm::APFloat &b) { return a + b; };
    // TODO: we could check overflows
    auto intOp = [](const llvm::APInt &a, const llvm::APInt &b) { return a + b; };
    if(auto res = constFoldBinaryOp<FloatAttr>(getType(), operands, floatOp))
        return res;
    if(auto res = constFoldBinaryOp<IntegerAttr>(getType(), operands, intOp))
        return res;
    return {};
}

mlir::OpFoldResult mlir::daphne::EwSubOp::fold(FoldAdaptor adaptor) {
    ArrayRef<Attribute> operands = adaptor.getOperands();
    auto floatOp = [](const llvm::APFloat &a, const llvm::APFloat &b) { return a - b; };
    auto intOp = [](const llvm::APInt &a, const llvm::APInt &b) { return a - b; };
    if(auto res = constFoldBinaryOp<FloatAttr>(getType(), operands, floatOp))
        return res;
    if(auto res = constFoldBinaryOp<IntegerAttr>(getType(), operands, intOp))
        return res;
    return {};
}

mlir::OpFoldResult mlir::daphne::EwMulOp::fold(FoldAdaptor adaptor) {
    ArrayRef<Attribute> operands = adaptor.getOperands();
    auto floatOp = [](const llvm::APFloat &a, const llvm::APFloat &b) { return a * b; };
    auto intOp = [](const llvm::APInt &a, const llvm::APInt &b) { return a * b; };
    if(auto res = constFoldBinaryOp<FloatAttr>(getType(), operands, floatOp))
        return res;
    if(auto res = constFoldBinaryOp<IntegerAttr>(getType(), operands, intOp))
        return res;
    return {};
}

mlir::OpFoldResult mlir::daphne::EwDivOp::fold(FoldAdaptor adaptor) {
    ArrayRef<Attribute> operands = adaptor.getOperands();
    auto floatOp = [](const llvm::APFloat &a, const llvm::APFloat &b) { return a / b; };
    auto sintOp = [&](const llvm::APInt &a, const llvm::APInt &b) {
        if(b == 0) {
            throw ErrorHandler::compilerError(
                this->getLoc(), "CanonicalizerPass (mlir::daphne::EwDivOp::fold)",
                "Can't divide by 0");
        }
        return a.sdiv(b);
    };
    auto uintOp = [&](const llvm::APInt &a, const llvm::APInt &b) {
        if(b == 0) {
            throw ErrorHandler::compilerError(
                this->getLoc(), "CanonicalizerPass (mlir::daphne::EwDivOp::fold)",
                "Can't divide by 0");
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

mlir::OpFoldResult mlir::daphne::EwPowOp::fold(FoldAdaptor adaptor) {
    ArrayRef<Attribute> operands = adaptor.getOperands();
    // TODO: EwPowOp integer constant folding
    auto floatOp = [](const llvm::APFloat &a, const llvm::APFloat &b) {
        return std::pow(a.convertToDouble(), b.convertToDouble());
    };
    if(auto res = constFoldBinaryOp<FloatAttr>(getType(), operands, floatOp))
        return res;
    return {};
}

mlir::OpFoldResult mlir::daphne::EwModOp::fold(FoldAdaptor adaptor) {
    ArrayRef<Attribute> operands = adaptor.getOperands();
    auto sintOp = [&](const llvm::APInt &a, const llvm::APInt &b) {
        if(b == 0) {
            throw ErrorHandler::compilerError(
                this->getLoc(), "CanonicalizerPass (mlir::daphne::EwModOp::fold)",
                "Can't compute mod 0");
        }
        return a.srem(b);
    };
    auto uintOp = [&](const llvm::APInt &a, const llvm::APInt &b) {
        if(b == 0) {
            throw ErrorHandler::compilerError(
                this->getLoc(), "CanonicalizerPass (mlir::daphne::EwModOp::fold)",
                "Can't compute mod 0");
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

mlir::OpFoldResult mlir::daphne::EwLogOp::fold(FoldAdaptor adaptor) {
    ArrayRef<Attribute> operands = adaptor.getOperands();
    auto floatOp = [](const llvm::APFloat &a, const llvm::APFloat &b) {
        // Compute the element-wise logarithm of a to the base b
        // Equivalent to log_b(a)
        return log(a.convertToDouble()) / log(b.convertToDouble());
    };
    if (auto res = constFoldBinaryOp<FloatAttr>(getType(), operands, floatOp))
        return res;
    return {};
}

mlir::OpFoldResult mlir::daphne::EwMinOp::fold(FoldAdaptor adaptor) {
    ArrayRef<Attribute> operands = adaptor.getOperands();
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

mlir::OpFoldResult mlir::daphne::EwMaxOp::fold(FoldAdaptor adaptor) {
    ArrayRef<Attribute> operands = adaptor.getOperands();
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

mlir::OpFoldResult mlir::daphne::EwAndOp::fold(FoldAdaptor adaptor) {
    ArrayRef<Attribute> operands = adaptor.getOperands();
    auto boolOp = [](const bool &a, const bool &b) { return a && b; };
    auto intOp = [](const llvm::APInt &a, const llvm::APInt &b) { return (a != 0) && (b != 0); };
    if(auto res = constFoldBinaryCmpOp<BoolAttr>(operands, boolOp))
        return res;
    // TODO: should output bool?
    if(auto res = constFoldBinaryOp<IntegerAttr>(getType(), operands, intOp))
        return res;
    return {};
}

mlir::OpFoldResult mlir::daphne::EwBitwiseAndOp::fold(FoldAdaptor adaptor) {
    return {};
}

mlir::OpFoldResult mlir::daphne::EwOrOp::fold(FoldAdaptor adaptor) {
    ArrayRef<Attribute> operands = adaptor.getOperands();
    auto boolOp = [](const bool &a, const bool &b) { return a || b; };
    auto intOp = [](const llvm::APInt &a, const llvm::APInt &b) { return (a != 0) || (b != 0); };
    if(auto res = constFoldBinaryCmpOp<BoolAttr>(operands, boolOp))
        return res;
    // TODO: should output bool
    if(auto res = constFoldBinaryOp<IntegerAttr>(getType(), operands, intOp))
        return res;
    return {};
}

mlir::OpFoldResult mlir::daphne::EwXorOp::fold(FoldAdaptor adaptor) {
    ArrayRef<Attribute> operands = adaptor.getOperands();
    auto boolOp = [](const bool &a, const bool &b) { return a ^ b; };
    auto intOp = [](const llvm::APInt &a, const llvm::APInt &b) { return (a != 0) ^ (b != 0); };
    if(auto res = constFoldBinaryCmpOp<BoolAttr>(operands, boolOp))
        return res;
    // TODO: should output bool
    if(auto res = constFoldBinaryOp<IntegerAttr>(getType(), operands, intOp))
        return res;
    return {};
}

mlir::OpFoldResult mlir::daphne::EwConcatOp::fold(FoldAdaptor adaptor) {
    ArrayRef<Attribute> operands = adaptor.getOperands();
    assert(operands.size() == 2 && "binary op takes two operands");
    if(!operands[0] || !operands[1])
        return {};

    if(llvm::isa<StringAttr>(operands[0]) && isa<StringAttr>(operands[1])) {
        auto lhs = operands[0].cast<StringAttr>();
        auto rhs = operands[1].cast<StringAttr>();

        auto concated = lhs.getValue().str() + rhs.getValue().str();
        return StringAttr::get(concated, getType());
    }
    return {};
}

// TODO This is duplicated from EwConcatOp. Actually, ConcatOp itself is only
// a temporary workaround, so it should be removed altogether later.
mlir::OpFoldResult mlir::daphne::ConcatOp::fold(FoldAdaptor adaptor) {
    ArrayRef<Attribute> operands = adaptor.getOperands();
    assert(operands.size() == 2 && "binary op takes two operands");
    if(!operands[0] || !operands[1])
        return {};

    if(llvm::isa<StringAttr>(operands[0]) && isa<StringAttr>(operands[1])) {
        auto lhs = operands[0].cast<StringAttr>();
        auto rhs = operands[1].cast<StringAttr>();

        auto concated = lhs.getValue().str() + rhs.getValue().str();
        return StringAttr::get(concated, getType());
    }
    return {};
}

mlir::OpFoldResult mlir::daphne::StringEqOp::fold(FoldAdaptor adaptor) {
    ArrayRef<Attribute> operands = adaptor.getOperands();
    assert(operands.size() == 2 && "binary op takes two operands");
    if (!operands[0] || !operands[1] || !llvm::isa<StringAttr>(operands[0]) ||
        !isa<StringAttr>(operands[1])) {
        return {};
    }

    auto lhs = operands[0].cast<StringAttr>();
    auto rhs = operands[1].cast<StringAttr>();

    return mlir::BoolAttr::get(getContext(), lhs.getValue() == rhs.getValue());
}

mlir::OpFoldResult mlir::daphne::EwEqOp::fold(FoldAdaptor adaptor) {
    ArrayRef<Attribute> operands = adaptor.getOperands();
    auto floatOp = [](const llvm::APFloat &a, const llvm::APFloat &b) { return a == b; };
    auto intOp = [](const llvm::APInt &a, const llvm::APInt &b) { return a == b; };
    // TODO: fix bool return
    if(auto res = constFoldBinaryOp<FloatAttr>(getType(), operands, floatOp))
        return res;
    if(auto res = constFoldBinaryOp<IntegerAttr>(getType(), operands, intOp))
        return res;
    return {};
}

mlir::OpFoldResult mlir::daphne::EwNeqOp::fold(FoldAdaptor adaptor) {
    ArrayRef<Attribute> operands = adaptor.getOperands();
    auto floatOp = [](const llvm::APFloat &a, const llvm::APFloat &b) { return a != b; };
    auto intOp = [](const llvm::APInt &a, const llvm::APInt &b) { return a != b; };
    // TODO: fix bool return
    if(auto res = constFoldBinaryOp<FloatAttr>(getType(), operands, floatOp))
        return res;
    if(auto res = constFoldBinaryOp<IntegerAttr>(getType(), operands, intOp))
        return res;
    return {};
}

mlir::OpFoldResult mlir::daphne::EwLtOp::fold(FoldAdaptor adaptor) {
    ArrayRef<Attribute> operands = adaptor.getOperands();
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

mlir::OpFoldResult mlir::daphne::EwLeOp::fold(FoldAdaptor adaptor) {
    ArrayRef<Attribute> operands = adaptor.getOperands();
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

mlir::OpFoldResult mlir::daphne::EwGtOp::fold(FoldAdaptor adaptor) {
    ArrayRef<Attribute> operands = adaptor.getOperands();
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

mlir::OpFoldResult mlir::daphne::EwGeOp::fold(FoldAdaptor adaptor) {
    ArrayRef<Attribute> operands = adaptor.getOperands();
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
 * @brief Transposition-aware matrix multiplication
 * Identifies if an input to a MatMulOp is the result of a TransposeOp; Rewrites the Operation,
 * passing transposition info as a flag, instead of transposing the matrix before multiplication
 */
mlir::LogicalResult mlir::daphne::MatMulOp::canonicalize(
        mlir::daphne::MatMulOp op, PatternRewriter &rewriter
) {    
    mlir::Value lhs = op.getLhs();
    mlir::Value rhs = op.getRhs();
    mlir::Value transa = op.getTransa();
    mlir::Value transb = op.getTransb();

    // TODO If transa or transb are not constant, we cannot continue on the respective side;
    // we cannot just assume false then.
    bool ta = CompilerUtils::constantOrDefault<bool>(transa, false);
    bool tb = CompilerUtils::constantOrDefault<bool>(transb, false);

    // TODO Turn on the transposition-awareness for the left-hand-side argument again (see #447).
    // mlir::daphne::TransposeOp lhsTransposeOp = lhs.getDefiningOp<mlir::daphne::TransposeOp>();
    mlir::daphne::TransposeOp rhsTransposeOp = rhs.getDefiningOp<mlir::daphne::TransposeOp>();

    //if (!lhsTransposeOp && !rhsTransposeOp){
    if (!rhsTransposeOp){
        return mlir::failure();
    }

#if 0
    // TODO Adapt PhyOperatorSelectionPass once this code is turned on again.
    if(lhsTransposeOp) {
        lhs = lhsTransposeOp.getArg();
        ta = !ta;
    }
#endif
    if(rhsTransposeOp) {
        rhs = rhsTransposeOp.getArg();
        tb = !tb;
    }

    rewriter.replaceOpWithNewOp<mlir::daphne::MatMulOp>(
        op, op.getType(), lhs, rhs,
        static_cast<mlir::Value>(rewriter.create<mlir::daphne::ConstantOp>(transa.getLoc(), ta)),
        static_cast<mlir::Value>(rewriter.create<mlir::daphne::ConstantOp>(transb.getLoc(), tb))
    );
    return mlir::success();
}

/**
 * @brief Replaces NumRowsOp by a constant, if the #rows of the input is known
 * (e.g., due to shape inference).
 */
mlir::LogicalResult mlir::daphne::NumRowsOp::canonicalize(
        mlir::daphne::NumRowsOp op, PatternRewriter &rewriter
) {
    ssize_t numRows = -1;
    
    mlir::Type inTy = op.getArg().getType();
    if(auto t = inTy.dyn_cast<mlir::daphne::MatrixType>())
        numRows = t.getNumRows();
    else if(auto t = inTy.dyn_cast<mlir::daphne::FrameType>())
        numRows = t.getNumRows();
    
    if(numRows != -1) {
        rewriter.replaceOpWithNewOp<mlir::daphne::ConstantOp>(
                op, rewriter.getIndexType(), rewriter.getIndexAttr(numRows)
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
    
    mlir::Type inTy = op.getArg().getType();
    if(auto t = inTy.dyn_cast<mlir::daphne::MatrixType>())
        numCols = t.getNumCols();
    else if(auto t = inTy.dyn_cast<mlir::daphne::FrameType>())
        numCols = t.getNumCols();
    
    if(numCols != -1) {
        rewriter.replaceOpWithNewOp<mlir::daphne::ConstantOp>(
                op, rewriter.getIndexType(), rewriter.getIndexAttr(numCols)
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
    
    mlir::Type inTy = op.getArg().getType();
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
                op, rewriter.getIndexType(), rewriter.getIndexAttr(numRows * numCols)
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
        mlir::daphne::ReadOp readOp = op.getMat().getDefiningOp<mlir::daphne::ReadOp>();
        if(!readOp || !readOp.getOperation()->hasOneUse())
            return mlir::failure();
        rewriter.replaceOp(
                op, {rewriter.create<mlir::daphne::DistributedReadOp>(
                        readOp.getLoc(), op.getType(), readOp.getFileName()
                )}
        );
        // TODO Instead of erasing the ReadOp here, the compiler should
        // generally remove unused SSA values. Then, we might even drop the
        // hasOneUse requirement above.
        rewriter.eraseOp(readOp);
        return mlir::success();
    }
};

// The EwBinarySca kernel does not handle string types in any way. In order to
// support simple string equivalence checks this canonicalizer rewrites the
// EwEqOp to the StringEqOp if one of the operands is of daphne::StringType.
mlir::LogicalResult mlir::daphne::EwEqOp::canonicalize(
    mlir::daphne::EwEqOp op, PatternRewriter &rewriter) {
    mlir::Value lhs = op.getLhs();
    mlir::Value rhs = op.getRhs();

    const bool lhsIsStr = llvm::isa<mlir::daphne::StringType>(lhs.getType());
    const bool rhsIsStr = llvm::isa<mlir::daphne::StringType>(rhs.getType());

    if (!lhsIsStr && !rhsIsStr) return mlir::failure();

    mlir::Type strTy = mlir::daphne::StringType::get(rewriter.getContext());
    if (!lhsIsStr)
        lhs = rewriter.create<mlir::daphne::CastOp>(op.getLoc(), strTy, lhs);
    if (!rhsIsStr)
        rhs = rewriter.create<mlir::daphne::CastOp>(op.getLoc(), strTy, rhs);

    rewriter.replaceOpWithNewOp<mlir::daphne::StringEqOp>(
        op, rewriter.getI1Type(), lhs, rhs);
    return mlir::success();
}

/**
 * @brief Replaces (1) `a + b` by `a concat b`, if `a` or `b` is a string,
 * and (2) `a + X` by `X + a` (`a` scalar, `X` matrix/frame).
 * 
 * (1) is important, since we use the `+`-operator for both addition and
 * string concatenation in DaphneDSL, while the types of the operands might be
 * known only after type inference.
 * 
 * (2) is important, since our kernels for elementwise binary operations only support
 * scalars as the right-hand-side operand so far (see #203).
 * 
 * @param op
 * @param rewriter
 * @return 
 */
mlir::LogicalResult mlir::daphne::EwAddOp::canonicalize(
        mlir::daphne::EwAddOp op, PatternRewriter &rewriter
) {
    mlir::Value lhs = op.getLhs();
    mlir::Value rhs = op.getRhs();

    const bool lhsIsStr = llvm::isa<mlir::daphne::StringType>(lhs.getType());
    const bool rhsIsStr = llvm::isa<mlir::daphne::StringType>(rhs.getType());
    if(lhsIsStr || rhsIsStr) {
        mlir::Type strTy = mlir::daphne::StringType::get(rewriter.getContext());
        if(!lhsIsStr)
            lhs = rewriter.create<mlir::daphne::CastOp>(op.getLoc(), strTy, lhs);
        if(!rhsIsStr)
            rhs = rewriter.create<mlir::daphne::CastOp>(op.getLoc(), strTy, rhs);
        rewriter.replaceOpWithNewOp<mlir::daphne::ConcatOp>(op, strTy, lhs, rhs);
        return mlir::success();
    }
    else {
        const bool lhsIsSca = !llvm::isa<mlir::daphne::MatrixType, mlir::daphne::FrameType>(lhs.getType());
        const bool rhsIsSca = !llvm::isa<mlir::daphne::MatrixType, mlir::daphne::FrameType>(rhs.getType());
        if(lhsIsSca && !rhsIsSca) {
            rewriter.replaceOpWithNewOp<mlir::daphne::EwAddOp>(op, op.getResult().getType(), rhs, lhs);
            return mlir::success();
        }
        return mlir::failure();
    }
}

/**
 * @brief Replaces `a - X` by `(X * -1) + a` (`a` scalar, `X` matrix/frame).
 * 
 * This is important, since our kernels for elementwise binary operations only support
 * scalars as the right-hand-side operand so far (see #203).
 * 
 * As a downside, an additional operation and intermediate result is introduced.
 * 
 * @param op
 * @param rewriter
 * @return 
 */
mlir::LogicalResult mlir::daphne::EwSubOp::canonicalize(
        mlir::daphne::EwSubOp op, PatternRewriter &rewriter
) {
    mlir::Value lhs = op.getLhs();
    mlir::Value rhs = op.getRhs();
    const bool lhsIsSca = !llvm::isa<mlir::daphne::MatrixType, mlir::daphne::FrameType>(lhs.getType());
    const bool rhsIsSca = !llvm::isa<mlir::daphne::MatrixType, mlir::daphne::FrameType>(rhs.getType());
    if(lhsIsSca && !rhsIsSca) {
        rewriter.replaceOpWithNewOp<mlir::daphne::EwAddOp>(
                op,
                op.getResult().getType(),
                rewriter.create<mlir::daphne::EwMulOp>(
                        op->getLoc(),
                        mlir::daphne::UnknownType::get(op->getContext()), // to be inferred
                        rhs,
                        rewriter.create<mlir::daphne::ConstantOp>(op->getLoc(), int64_t(-1))
                ),
                lhs
        );
        return mlir::success();
    }
    return mlir::failure();
}

/**
 * @brief Replaces `a * X` by `X * a` (`a` scalar, `X` matrix/frame).
 * 
 * This is important, since our kernels for elementwise binary operations only support
 * scalars as the right-hand-side operand so far (see #203).
 * 
 * @param op
 * @param rewriter
 * @return 
 */
mlir::LogicalResult mlir::daphne::EwMulOp::canonicalize(
        mlir::daphne::EwMulOp op, PatternRewriter &rewriter
) {
    mlir::Value lhs = op.getLhs();
    mlir::Value rhs = op.getRhs();
    const bool lhsIsSca = !llvm::isa<mlir::daphne::MatrixType, mlir::daphne::FrameType>(lhs.getType());
    const bool rhsIsSca = !llvm::isa<mlir::daphne::MatrixType, mlir::daphne::FrameType>(rhs.getType());
    if(lhsIsSca && !rhsIsSca) {
        rewriter.replaceOpWithNewOp<mlir::daphne::EwMulOp>(op, op.getResult().getType(), rhs, lhs);
        return mlir::success();
    }
    return mlir::failure();
}

/**
 * @brief Replaces `a / X` by `(X ^ -1) * a` (`a` scalar, `X` matrix/frame),
 * if `X` has a floating-point value type.
 * 
 * This is important, since our kernels for elementwise binary operations only support
 * scalars as the right-hand-side operand so far (see #203).
 * 
 * As a downside, an additional operation and intermediate result is introduced.
 * 
 * @param op
 * @param rewriter
 * @return 
 */
mlir::LogicalResult mlir::daphne::EwDivOp::canonicalize(
        mlir::daphne::EwDivOp op, PatternRewriter &rewriter
) {
    mlir::Value lhs = op.getLhs();
    mlir::Value rhs = op.getRhs();
    const bool lhsIsSca = !llvm::isa<mlir::daphne::MatrixType, mlir::daphne::FrameType>(lhs.getType());
    const bool rhsIsSca = !llvm::isa<mlir::daphne::MatrixType, mlir::daphne::FrameType>(rhs.getType());
    const bool rhsIsFP = llvm::isa<mlir::FloatType>(CompilerUtils::getValueType(rhs.getType()));
    if(lhsIsSca && !rhsIsSca && rhsIsFP) {
        rewriter.replaceOpWithNewOp<mlir::daphne::EwMulOp>(
                op,
                op.getResult().getType(),
                rewriter.create<mlir::daphne::EwPowOp>(
                        op->getLoc(),
                        mlir::daphne::UnknownType::get(op->getContext()), // to be inferred
                        rhs,
                        rewriter.create<mlir::daphne::ConstantOp>(op->getLoc(), double(-1))
                ),
                lhs
        );
        return mlir::success();
    }
    return mlir::failure();
}

void mlir::daphne::DistributeOp::getCanonicalizationPatterns(
        RewritePatternSet &results, MLIRContext *context
) {
    results.add<SimplifyDistributeRead>(context);
}

mlir::LogicalResult mlir::daphne::CondOp::canonicalize(mlir::daphne::CondOp op,
                                                       mlir::PatternRewriter &rewriter)
{
    mlir::Value cond = op.getCond();
    if(llvm::isa<mlir::daphne::UnknownType, mlir::daphne::MatrixType, mlir::daphne::FrameType>(cond.getType()))
        // If the condition is not a scalar, we cannot rewrite the operation here.
        return mlir::failure();
    else {
        // If the condition is a scalar, we rewrite the operation to an if-then-else construct
        // using the SCF dialect.
        // TODO Check if it is really a scalar.

        mlir::Location loc = op.getLoc();

        // Ensure that the condition is a boolean.
        if(!cond.getType().isSignlessInteger(1))
            cond = rewriter.create<mlir::daphne::CastOp>(loc, rewriter.getI1Type(), cond);

        mlir::Block thenBlock;
        mlir::Block elseBlock;
        mlir::Value thenVal = op.getThenVal();
        mlir::Value elseVal = op.getElseVal();

        // Get rid of frame column labels, since they interfere with the type comparison (see #485).
        if(auto thenFrmTy = thenVal.getType().dyn_cast<daphne::FrameType>())
            if(thenFrmTy.getLabels() != nullptr)
                thenVal = rewriter.create<mlir::daphne::CastOp>(loc, thenFrmTy.withLabels(nullptr), thenVal);
        if(auto elseFrmTy = elseVal.getType().dyn_cast<daphne::FrameType>())
            if(elseFrmTy.getLabels() != nullptr)
                elseVal = rewriter.create<mlir::daphne::CastOp>(loc, elseFrmTy.withLabels(nullptr), elseVal);

        // Check if the types of the then-value and the else-value are the same.
        if(thenVal.getType() != elseVal.getType()) {
            if(llvm::isa<daphne::UnknownType>(thenVal.getType()) || llvm::isa<daphne::UnknownType>(elseVal.getType()))
                // If one of them is unknown, we abort the rewrite (but this is not an error).
                // The type may become known later, this rewrite will be triggered again.
                return mlir::failure();
            else
                // If both types are known, but different, this is an error.
                // TODO We could try to cast the types.
                throw ErrorHandler::compilerError(
                    op, "CanonicalizerPass (mlir::daphne::CondOp)",
                    "the then/else-values of CondOp must have the same value "
                    "type");
        }

        {
            // Save the insertion point (automatically restored at the end of the block).
            PatternRewriter::InsertionGuard insertGuard(rewriter);

            // TODO The current implementation only makes sure that the correct value is
            // returned, but the operations calculating the then/else-values are still
            // outside the if-then-else and will always both be executed (unless, e.g.,
            // the entire branching can be elimitated). This could be good (e.g., if
            // the then/else-values have common subexpressions with other code) or bad
            // (e.g., if they are expensive to compute). See #486.

            // Create yield-operations in both branches.
            rewriter.setInsertionPointToEnd(&thenBlock);
            rewriter.create<mlir::scf::YieldOp>(loc, thenVal);
            rewriter.setInsertionPointToEnd(&elseBlock);
            rewriter.create<mlir::scf::YieldOp>(loc, elseVal);
        }

        // Helper functions to move the operations in the two blocks created above
        // into the actual branches of the if-operation.
        auto insertThenBlockDo = [&](mlir::OpBuilder & nested, mlir::Location loc) {
            nested.getBlock()->getOperations().splice(nested.getBlock()->end(), thenBlock.getOperations());
        };
        auto insertElseBlockDo = [&](mlir::OpBuilder & nested, mlir::Location loc) {
            nested.getBlock()->getOperations().splice(nested.getBlock()->end(), elseBlock.getOperations());
        };

        // Replace the daphne::CondOp by an scf::IfOp.
        rewriter.replaceOpWithNewOp<mlir::scf::IfOp>(
            op, cond, insertThenBlockDo, insertElseBlockDo
        );

        return mlir::success();
    }
}

mlir::LogicalResult mlir::daphne::ConvertDenseMatrixToMemRef::canonicalize(
    mlir::daphne::ConvertDenseMatrixToMemRef op,
    mlir::PatternRewriter &rewriter) {
    // removes unnecessary conversions of MemRef -> DM -> MemRef
    mlir::Operation *dmNode = op->getOperand(0).getDefiningOp();

    if (!llvm::isa<mlir::daphne::ConvertMemRefToDenseMatrix>(dmNode))
        return failure();

    mlir::Operation *originalMemRefOp =
        dmNode->getPrevNode()->getOperand(0).getDefiningOp();
    op.replaceAllUsesWith(originalMemRefOp);

    rewriter.eraseOp(op);
    if (dmNode->getUsers().empty()) rewriter.eraseOp(dmNode);

    return mlir::success();
}

mlir::LogicalResult mlir::daphne::ConvertMemRefToDenseMatrix::canonicalize(
    mlir::daphne::ConvertMemRefToDenseMatrix op,
    mlir::PatternRewriter &rewriter) {
    mlir::Operation *extractPtr = op->getPrevNode();
    auto srcMemRef = extractPtr->getOperand(0).getDefiningOp();
    extractPtr->moveAfter(srcMemRef);
    op->moveAfter(extractPtr);

    return mlir::success();
}

