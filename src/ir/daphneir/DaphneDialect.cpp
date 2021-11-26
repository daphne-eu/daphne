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
    if (keyword == "Matrix") {
        ssize_t numRows = -1;
        ssize_t numCols = -1;
        double sparsity = -1.0;
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
        if (succeeded(parser.parseOptionalColon())) {
            while (failed(parser.parseOptionalGreater())) {
                if (succeeded(parser.parseKeyword("sp"))) {
                    if (sparsity != -1.0) {
                        // read sparsity twice
                        return nullptr;
                    }
                    if (parser.parseLSquare() || parser.parseFloat(sparsity) || parser.parseRSquare()) {
                        return nullptr;
                    }
                }
                else {
                    return nullptr;
                }
            }
        }
        else if(parser.parseGreater()) {
            return nullptr;
        }

        return MatrixType::get(
                parser.getBuilder().getContext(), elementType, numRows, numCols, -1.0
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
        if (sparsity != -1.0) {
            os << ":sp[" << sparsity << ']';
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

namespace mlir::daphne {
    namespace detail {
        struct MatrixTypeStorage : public ::mlir::TypeStorage {
            // TODO: adapt epsilon for equality check (I think the only use is saving memory for the MLIR-IR representation of this type)
            //  the choosen epsilon directly defines how accurate our sparsity inference can be
            constexpr static const float epsilon = 1e-6;
            MatrixTypeStorage(::mlir::Type elementType, ssize_t numRows, ssize_t numCols, float sparsity)
                : elementType(elementType), numRows(numRows), numCols(numCols), sparsity(sparsity) {}

            /// The hash key is a tuple of the parameter types.
            using KeyTy = std::tuple<::mlir::Type, ssize_t, ssize_t, float>;
            bool operator==(const KeyTy &tblgenKey) const {
                if(!(elementType == std::get<0>(tblgenKey)))
                    return false;
                if(numRows != std::get<1>(tblgenKey))
                    return false;
                if(numCols != std::get<2>(tblgenKey))
                    return false;
                if(std::fabs(sparsity - std::get<3>(tblgenKey)) >= epsilon)
                    return false;
                return true;
            }
            static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
                auto float_hashable = static_cast<ssize_t>(std::get<3>(tblgenKey) / epsilon);
                return ::llvm::hash_combine(std::get<0>(tblgenKey),
                    std::get<1>(tblgenKey),
                    std::get<2>(tblgenKey),
                    float_hashable);
            }

            /// Define a construction method for creating a new instance of this
            /// storage.
            static MatrixTypeStorage *construct(::mlir::TypeStorageAllocator &allocator,
                                                const KeyTy &tblgenKey) {
                auto elementType = std::get<0>(tblgenKey);
                auto numRows = std::get<1>(tblgenKey);
                auto numCols = std::get<2>(tblgenKey);
                auto sparsity = std::get<3>(tblgenKey);

                return new(allocator.allocate<MatrixTypeStorage>())
                    MatrixTypeStorage(elementType, numRows, numCols, sparsity);
            }
            ::mlir::Type elementType;
            ssize_t numRows;
            ssize_t numCols;
            float sparsity;
        };
    }
    ::mlir::Type MatrixType::getElementType() const { return getImpl()->elementType; }
    ssize_t MatrixType::getNumRows() const { return getImpl()->numRows; }
    ssize_t MatrixType::getNumCols() const { return getImpl()->numCols; }
    float MatrixType::getSparsity() const { return getImpl()->sparsity; }
}

mlir::OpFoldResult mlir::daphne::ConstantOp::fold(mlir::ArrayRef<mlir::Attribute> operands)
{
    assert(operands.empty() && "constant has no operands");
    return value();
}

::mlir::LogicalResult mlir::daphne::MatrixType::verify(
        ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
        Type elementType,
        ssize_t numRows, ssize_t numCols, float sparsity
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
    for (auto e : llvm::enumerate(resultsToReplace)) {
        auto resultToReplace = e.value();
        auto i = e.index();
        resultToReplace.replaceAllUsesWith(pipelineOp.getResult(i));
    }
    op.erase();
    return success();
}
