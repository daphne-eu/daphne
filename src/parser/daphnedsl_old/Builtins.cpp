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

#include "parser/daphnedsl/Builtins.h"

#include <utility>

using namespace mlir;
using namespace mlir::daphne;

template <typename T>
Builtin<T>::Builtin(std::vector<unsigned int> expectedNumOfParams)
    : expectedNumOfParams(std::move(expectedNumOfParams)) {}

template <typename T> LogicalResult Builtin<T>::checkNumParams(Location &loc, llvm::StringRef name, size_t size) {
    if (std::find(expectedNumOfParams.begin(), expectedNumOfParams.end(), size) == expectedNumOfParams.end()) {
        emitError(loc) << '`' << name << "` does not accept " << size << " parameters\n";
        return failure();
    }
    return success();
}

template <typename T> Builtin<T>::~Builtin() = default;

const llvm::StringRef PrintBuiltin::name = "print";

PrintOp PrintBuiltin::create(OpBuilder builder, Location &loc, ValueRange values) {
    if (failed(checkNumParams(loc, name, values.size())))
        return nullptr;
    return builder.create<PrintOp>(loc, values[0]);
}

const llvm::StringRef RandBuiltin::name = "rand";

RandMatrixOp RandBuiltin::create(OpBuilder builder, Location &loc, ValueRange values) {
    if (values.size() == 6) {
        return builder.create<RandMatrixOp>(loc, MatrixType::get(builder.getContext(), values[2].getType()), values[0],
                                            values[1], values[2], values[3], values[4], values[5]);
    }
    if (failed(checkNumParams(loc, name, values.size())))
        return nullptr;
    llvm_unreachable("Number of operands should be checked or handled");
}

const llvm::StringRef TransposeBuiltin::name = "t";

TransposeOp TransposeBuiltin::create(OpBuilder builder, Location &loc, ValueRange values) {
    if (failed(checkNumParams(loc, name, values.size())))
        return nullptr;
    return builder.create<TransposeOp>(loc, values[0]);
}

antlrcpp::Any Builtins::build(OpBuilder &builder, Location &loc, ValueRange values, const std::string &name) {
    if (name == PrintBuiltin::name) {
        Operation *op = PrintBuiltin().create(builder, loc, values);
        return op;
    }
    if (name == RandBuiltin::name) {
        Value rand = RandBuiltin().create(builder, loc, values);
        return rand;
    }
    if (name == TransposeBuiltin::name) {
        Value transpose = TransposeBuiltin().create(builder, loc, values);
        return transpose;
    }
    //    if (name == "sum") {
    //        if (values[0].getType().isa<daphne::MatrixType>()) {
    //            Value sum = builder.create<SumOp>(loc,
    //            values[0].getType().dyn_cast<daphne::MatrixType>().getElementType(),
    //            values[0]); return sum;
    //        }
    //        else
    //            // TODO This is probably not a good way of handling it.
    //            return nullptr;
    //    }
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
    //    Value repeat = builder.create<RepeatOp>(loc, values[0], values[1],
    //    values[2]); return repeat;
    //  }
    return nullptr;
}
