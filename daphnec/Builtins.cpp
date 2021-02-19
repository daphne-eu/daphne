#include "Builtins.h"

#include <utility>

using namespace mlir;
using namespace mlir::daphne;

template<typename T>
Builtin<T>::Builtin(std::vector<unsigned int> expectedNumOfParams)
: expectedNumOfParams(std::move(expectedNumOfParams))
{
}

template<typename T>
LogicalResult Builtin<T>::checkNumParams(Location &loc, llvm::StringRef name, size_t size)
{
    if (std::find(expectedNumOfParams.begin(), expectedNumOfParams.end(), size) == expectedNumOfParams.end()) {
        emitError(loc) << '`' << name << "` does not accept " << size << " parameters\n";
        return failure();
    }
    return success();
}

template<typename T>
Builtin<T>::~Builtin() = default;

const llvm::StringRef PrintBuiltin::name = "print";

PrintOp PrintBuiltin::create(OpBuilder builder,
                             Location &loc,
                             ValueRange values)
{
    if (failed(checkNumParams(loc, name, values.size())))
        return nullptr;
    return builder.create<PrintOp>(loc, values[0]);
}

const llvm::StringRef RandBuiltin::name = "rand";

RandOp RandBuiltin::create(OpBuilder builder,
                           Location &loc,
                           ValueRange values)
{
    if (values.size() == 4) {
        return builder.create<RandOp>(
                loc,
                MatrixType::get(builder.getContext(), builder.getF64Type()),
                values[0], values[1], values[2], values[3]
                );
    }
    if (failed(checkNumParams(loc, name, values.size())))
        return nullptr;
    llvm_unreachable("Number of operands should be checked or handled");
}

const llvm::StringRef TransposeBuiltin::name = "t";

TransposeOp TransposeBuiltin::create(OpBuilder builder, Location &loc, ValueRange values)
{
    if (failed(checkNumParams(loc, name, values.size())))
        return nullptr;
    return builder.create<TransposeOp>(loc, values[0]);
}

antlrcpp::Any Builtins::build(OpBuilder &builder,
                              Location &loc,
                              ValueRange values,
                              const std::string &name)
{
    if (name == PrintBuiltin::name) {
        Operation *op = PrintBuiltin().create(builder, loc, values);
        return op;
    }
    if (name == RandBuiltin::name) {
        RandBuiltin p;
        Value rand = RandBuiltin().create(builder, loc, values);
        return rand;
    }
    if (name == TransposeBuiltin::name) {
        Value transpose = TransposeBuiltin().create(builder, loc, values);
        return transpose;
    }
    //  if (name == "sum") {
    //    Value sum = builder.create<SumOp>(loc, values[0]);
    //    return sum;
    //  }
    //  if (name == "rowSums") {
    //    Value rowSums = builder.create<RowAggOp>(loc, AggFn::sum, values[0]);
    //    return rowSums;
    //  }
    //  if (name == "rowMins") {
    //    Value rowMins = builder.create<RowAggOp>(loc, AggFn::min, values[0]);
    //    return rowMins;
    //  }
    //  if (name == "colSums") {
    //    Value colSums = builder.create<ColAggOp>(loc, AggFn::sum, values[0]);
    //    return colSums;
    //  }
    //  if (name == "colMins") {
    //    Value colMins = builder.create<ColAggOp>(loc, AggFn::min, values[0]);
    //    return colMins;
    //  }
    //  if (name == "repeat") {
    //    Value repeat = builder.create<RepeatOp>(loc, values[0], values[1], values[2]);
    //    return repeat;
    //  }
    return nullptr;
}
