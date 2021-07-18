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
#define GET_OP_CLASSES
#include <ir/daphneir/DaphneOps.cpp.inc>
#define GET_TYPEDEF_CLASSES
#include <ir/daphneir/DaphneOpsTypes.cpp.inc>

#include "mlir/IR/Builders.h"
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
        mlir::Type elementType;
        llvm::SMLoc locElementType;
        if (parser.parseLess() || parser.getCurrentLocation(&locElementType)
            || parser.parseType(elementType) || parser.parseGreater()) {
            return nullptr;
        }

        // TODO: check valid element types
        return MatrixType::get(parser.getBuilder().getContext(), elementType);
    }
    else if (keyword == "Frame") {
        if (parser.parseLess() || parser.parseLSquare()) {
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
        return FrameType::get(parser.getBuilder().getContext(), cts);
    }
    else {
        parser.emitError(parser.getCurrentLocation()) << "Parsing failed, keyword `" << keyword << "` not recognized!";
        return nullptr;
    }
}

void mlir::daphne::DaphneDialect::printType(mlir::Type type,
                                            mlir::DialectAsmPrinter &os) const
{
    if (auto t = type.dyn_cast<mlir::daphne::MatrixType>())
        os << "Matrix<" << t.getElementType() << '>';
    else if (auto t = type.dyn_cast<mlir::daphne::FrameType>()) {
        std::vector<mlir::Type> cts = t.getColumnTypes();
        os << "Frame<[";
        if(!cts.empty()) {
            os << cts[0];
            for (size_t i = 1; i < cts.size(); i++)
                os << ", " << cts[i];
        }
        os << "]>";
    }
    else if (type.isa<mlir::daphne::StringType>())
        os << "String";
    else if (auto t = type.dyn_cast<mlir::daphne::VariadicPackType>())
        os << "VariadicPack<" << t.getContainedType() << '>';
};

mlir::OpFoldResult mlir::daphne::ConstantOp::fold(mlir::ArrayRef<mlir::Attribute> operands)
{
    assert(operands.empty() && "constant has no operands");
    return value();
}

::mlir::LogicalResult mlir::daphne::MatrixType::verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type elementType)
{
    if (elementType.isSignedInteger(64) || elementType.isF64() || elementType.isIndex())
        return mlir::success();
    else
        return emitError() << "invalid matrix element type: " << elementType;
}

::mlir::LogicalResult mlir::daphne::FrameType::verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, std::vector<Type> columnTypes)
{
    return mlir::success();
}
