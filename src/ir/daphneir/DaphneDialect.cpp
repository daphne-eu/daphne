#include <mlir/Dialect/daphne/Daphne.h>
#define GET_OP_CLASSES
#include <mlir/Dialect/daphne/DaphneOps.cpp.inc>

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
#include <mlir/Dialect/daphne/DaphneOps.cpp.inc>
            >();
    addTypes<
#define GET_TYPEDEF_LIST
#include <mlir/Dialect/daphne/DaphneOpsTypes.cpp.inc>
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
    return mlir::IntegerType();
};

void mlir::daphne::DaphneDialect::printType(mlir::Type type,
                                            mlir::DialectAsmPrinter &os) const
{
    if (type.isa<mlir::daphne::MatrixType>())
        os << "Matrix<" << type.dyn_cast<mlir::daphne::MatrixType>().getElementType() << '>';
    else if (type.isa<mlir::daphne::FrameType>()) {
        std::vector<mlir::Type> cts = type.dyn_cast<mlir::daphne::FrameType>().getColumnTypes();
        os << "Frame<[" << cts[0];
        for (size_t i = 1; i < cts.size(); i++)
            os << ", " << cts[i];
        os << "]>";
    }
};

mlir::OpFoldResult mlir::daphne::ConstantOp::fold(mlir::ArrayRef<mlir::Attribute> operands)
{
    assert(operands.empty() && "constant has no operands");
    return value();
}

::mlir::LogicalResult mlir::daphne::MatrixType::verifyConstructionInvariants(::mlir::Location loc, Type elementType)
{
    if (elementType.isSignedInteger(64) || elementType.isF64())
        return mlir::success();
    else
        return mlir::emitError(loc) << "invalid matrix element type: " << elementType;
}

::mlir::LogicalResult mlir::daphne::FrameType::verifyConstructionInvariants(::mlir::Location loc, std::vector<Type> columnTypes)
{
    return mlir::success();
}
