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

#include <parser/daphnedsl/DaphneDSLBuiltins.h>
#include <ir/daphneir/Daphne.h>

#include "antlr4-runtime.h"

#include <stdexcept>
#include <string>
#include <vector>

#include <cstdlib>

// ************************************************************************
// Checking number of arguments
// ************************************************************************

void DaphneDSLBuiltins::checkNumArgsExact(const std::string & func, size_t numArgs, size_t numArgsExact) {
    if(numArgs != numArgsExact)
        throw std::runtime_error(
                "built-in function '" + func + "' expects exactly " +
                std::to_string(numArgsExact) + " argument(s), but got " +
                std::to_string(numArgs)
        );
}

void DaphneDSLBuiltins::checkNumArgsBetween(const std::string & func, size_t numArgs, size_t numArgsMin, size_t numArgsMax) {
    if(numArgs < numArgsMin || numArgs > numArgsMax)
        throw std::runtime_error(
                "built-in function '" + func + "' expects between " +
                std::to_string(numArgsMin) + " and " + std::to_string(numArgsMax) +
                " argument(s), but got " + std::to_string(numArgs)
        );
}

void DaphneDSLBuiltins::checkNumArgsMin(const std::string & func, size_t numArgs, size_t numArgsMin) {
    if(numArgs < numArgsMin)
        throw std::runtime_error(
                "built-in function '" + func + "' at least " +
                std::to_string(numArgsMin) + " argument(s), but got " +
                std::to_string(numArgs)
        );
}

void DaphneDSLBuiltins::checkNumArgsEven(const std::string & func, size_t numArgs) {
    if(numArgs % 2)
        throw std::runtime_error(
                "built-in function '" + func + 
                "' expects an even number of arguments, but got " + 
                std::to_string(numArgs)
        );
}

// ************************************************************************
// Creating similar DaphneIR operations
// ************************************************************************

template<class NumOp>
mlir::Value DaphneDSLBuiltins::createNumOp(mlir::Location loc, const std::string & func, const std::vector<mlir::Value> & args) {
    checkNumArgsExact(func, args.size(), 1);
    return static_cast<mlir::Value>(builder.create<NumOp>(
            loc, utils.sizeType, args[0]
    ));
}

template<class EwUnaryOp>
mlir::Value DaphneDSLBuiltins::createEwUnaryOp(mlir::Location loc, const std::string & func, const std::vector<mlir::Value> & args) {
    checkNumArgsExact(func, args.size(), 1);
    return static_cast<mlir::Value>(builder.create<EwUnaryOp>(
            loc, args[0].getType(), args[0]
    ));
}

template<class EwBinaryOp>
mlir::Value DaphneDSLBuiltins::createEwBinaryOp(mlir::Location loc, const std::string & func, const std::vector<mlir::Value> & args) {
    checkNumArgsExact(func, args.size(), 2);
    return static_cast<mlir::Value>(builder.create<EwBinaryOp>(
            loc, args[0], args[1]
    ));
}

template<class RowAggOp, class ColAggOp>
mlir::Value DaphneDSLBuiltins::createRowOrColAggOp(mlir::Location loc, const std::string & func, const std::vector<mlir::Value> & args) {
    checkNumArgsExact(func, args.size(), 2);
    if(auto co = args[1].getDefiningOp<mlir::daphne::ConstantOp>()) {
        llvm::APInt axis = co.value().dyn_cast<mlir::IntegerAttr>().getValue();
        if(axis == 0) 
            return static_cast<mlir::Value>(
                    builder.create<RowAggOp>(
                            loc, args[0].getType(), args[0]
                    )
            );
        else if(axis == 1)
            return static_cast<mlir::Value>(
                    builder.create<ColAggOp>(
                            loc, args[0].getType(), args[0]
                    )
            );
        else
            throw std::runtime_error("invalid axis");
    }
    else
        throw std::runtime_error("second argument of aggregation must be a constant");
}

template<class GrpAggOp>
mlir::Value DaphneDSLBuiltins::createGrpAggOp(mlir::Location loc, const std::string & func, const std::vector<mlir::Value> & args) {
    checkNumArgsExact(func, args.size(), 3);
    mlir::Value arg = args[0];
    mlir::Value groupIds = args[1];
    mlir::Value numGroups = utils.castSizeIf(args[2]);
    return static_cast<mlir::Value>(builder.create<GrpAggOp>(
            loc, args[0].getType(), arg, groupIds, numGroups
    ));
}

template<class AllAggOp, class RowAggOp, class ColAggOp, class GrpAggOp>
mlir::Value DaphneDSLBuiltins::createAnyAggOp(mlir::Location loc, const std::string & func, const std::vector<mlir::Value> & args) {
    const size_t numArgs = args.size();
    checkNumArgsBetween(func, numArgs, 1, 3);
    if(args.size() == 1)
        return static_cast<mlir::Value>(
                builder.create<AllAggOp>(
                        // TODO this crashes if the input is not a matrix
                        loc, args[0].getType().dyn_cast<mlir::daphne::MatrixType>().getElementType(), args[0]
                )
        );
    else if(numArgs == 2)
        return createRowOrColAggOp<RowAggOp, ColAggOp>(loc, func, args);
    else // numArgs == 3
        return createGrpAggOp<GrpAggOp>(loc, func, args);
}

template<class CumAggOp>
mlir::Value DaphneDSLBuiltins::createCumAggOp(mlir::Location loc, const std::string & func, const std::vector<mlir::Value> & args) {
    checkNumArgsExact(func, args.size(), 1);
    return static_cast<mlir::Value>(builder.create<CumAggOp>(
            loc, args[0].getType(), args[0]
    ));
}

template<class BindOp>
mlir::Value DaphneDSLBuiltins::createBindOp(mlir::Location loc, const std::string & func, const std::vector<mlir::Value> & args) {
    checkNumArgsExact(func, args.size(), 2);
    return static_cast<mlir::Value>(builder.create<BindOp>(
            loc, args[0].getType(), args[0], args[1]
    ));
}

template<class TheOp>
mlir::Value DaphneDSLBuiltins::createSameTypeUnaryOp(mlir::Location loc, const std::string & func, const std::vector<mlir::Value> & args) {
    checkNumArgsExact(func, args.size(), 1);
    return static_cast<mlir::Value>(builder.create<TheOp>(
            loc, args[0].getType(), args[0]
    ));
}

mlir::Value DaphneDSLBuiltins::createTriOp(mlir::Location loc, const std::string & func, const std::vector<mlir::Value> & args, bool upper) {
    checkNumArgsExact(func, args.size(), 3);
    mlir::Value arg = args[0];
    mlir::Value upper2 = builder.create<mlir::daphne::ConstantOp>(loc, builder.getIntegerAttr(builder.getI1Type(), upper));
    mlir::Value diag = utils.castBoolIf(args[1]);
    mlir::Value values = utils.castBoolIf(args[2]);
    return static_cast<mlir::Value>(builder.create<mlir::daphne::TriOp>(
            loc, arg.getType(), arg, upper2, diag, values
    ));
}

template<class SetOp>
mlir::Value DaphneDSLBuiltins::createSetOp(mlir::Location loc, const std::string & func, const std::vector<mlir::Value> & args) {
    checkNumArgsExact(func, args.size(), 2);
    return static_cast<mlir::Value>(builder.create<SetOp>(
            loc, args[0].getType(), args[0], args[1]
    ));
}

template<class JoinOp>
mlir::Value DaphneDSLBuiltins::createJoinOp(mlir::Location loc, const std::string & func, const std::vector<mlir::Value> & args) {
    const size_t numArgs = args.size();
    checkNumArgsMin(func, numArgs, 4);
    checkNumArgsEven(func, numArgs);
    mlir::Value lhs = args[0];
    mlir::Value rhs = args[1];
    std::vector<mlir::Value> leftOn;
    std::vector<mlir::Value> rightOn;
    const size_t numCols = (numArgs - 2) / 2;
    for(size_t i = 0; i < numCols; i++) {
        leftOn.push_back(utils.castSizeIf(args[2 + i]));
        rightOn.push_back(utils.castSizeIf(args[2 + numCols + i]));
    }
    std::vector<mlir::Type> colTypes;
    for(mlir::Type t : lhs.getType().dyn_cast<mlir::daphne::FrameType>().getColumnTypes())
        colTypes.push_back(t);
    for(mlir::Type t : rhs.getType().dyn_cast<mlir::daphne::FrameType>().getColumnTypes())
        colTypes.push_back(t);
    mlir::Type t = mlir::daphne::FrameType::get(builder.getContext(), colTypes);
    return static_cast<mlir::Value>(builder.create<JoinOp>(
            loc, t, lhs, rhs, leftOn, rightOn
    ));
}

antlrcpp::Any DaphneDSLBuiltins::build(mlir::Location loc, const std::string & func, const std::vector<mlir::Value> & args) {
    using namespace mlir::daphne;

    const size_t numArgs = args.size();
    
    // Basically, for each DaphneDSL built-in function we need to:
    // - check if the number of provided arguments is valid
    // - if required by the DaphneIR operation to be created: ensure that the
    //   arguments have the right types (by inserting DaphneIR casts)
    // - create and return the DaphneIR operation represented by the DaphneDSL
    //   built-in function (including the inference of the result type)
    // TODO Outsource the result type inference (issue #44) in an MLIR way.
    
    // Note that some built-in functions require some more advanced steps.
    
    // There are several groups of DaphneIR operations that are very similar
    // w.r.t. their arguments and results; so there are specific template
    // functions for creating them.

    // ********************************************************************
    // Data generation
    // ********************************************************************

    if(func == "fill") {
        checkNumArgsExact(func, numArgs, 3);
        mlir::Value arg = args[0];
        mlir::Value numRows = utils.castSizeIf(args[1]);
        mlir::Value numCols = utils.castSizeIf(args[2]);
        return static_cast<mlir::Value>(builder.create<FillOp>(
                loc, utils.matrixOf(arg), arg, numRows, numCols
        ));
    }
    if(func == "frame") {
        checkNumArgsMin(func, numArgs, 1);
        std::vector<mlir::Type> colTypes;
        for(auto arg : args)
            colTypes.push_back(arg.getType().dyn_cast<MatrixType>().getElementType());
        mlir::Type t = FrameType::get(builder.getContext(), colTypes);
        return static_cast<mlir::Value>(builder.create<CreateFrameOp>(loc, t, args));
    }
    if(func == "diagMatrix")
        return createSameTypeUnaryOp<DiagMatrixOp>(loc, func, args);
    if(func == "rand") {
        checkNumArgsExact(func, numArgs, 6);
        mlir::Value numRows = utils.castSizeIf(args[0]);
        mlir::Value numCols = utils.castSizeIf(args[1]);
        mlir::Value min = args[2];
        mlir::Value max = args[3];
        mlir::Value sparsity = utils.castIf(builder.getF64Type(), args[4]);
        mlir::Value seed = utils.castSeedIf(args[5]);
        return static_cast<mlir::Value>(builder.create<RandMatrixOp>(
                loc,
                MatrixType::get(builder.getContext(), min.getType()),
                numRows, numCols, min, max, sparsity, seed
        ));
    }
    if(func == "sample") {
        checkNumArgsExact(func, numArgs, 4);
        mlir::Value range = utils.castSizeIf(args[0]);
        mlir::Value size = utils.castSizeIf(args[1]);
        mlir::Value withReplacement = utils.castBoolIf(args[2]);
        mlir::Value seed = utils.castSeedIf(args[3]);
        return static_cast<mlir::Value>(
                builder.create<SampleOp>(
                        loc, utils.matrixOfSizeType, range, size, withReplacement, seed
                )
        );
    }
    if(func == "seq") {
        checkNumArgsExact(func, numArgs, 3);
        mlir::Value from = args[0];
        mlir::Value to = args[1];
        mlir::Value inc= args[2];
        return static_cast<mlir::Value>(
                builder.create<SeqOp>(
                        loc, utils.matrixOf(from), from, to, inc
                )
        );
    }

    // ********************************************************************
    // Matrix/frame dimensions
    // ********************************************************************

    if(func == "nrow")
        return createNumOp<NumRowsOp>(loc, func, args);
    if(func == "ncol")
        return createNumOp<NumColsOp>(loc, func, args);
    if(func == "ncell")
        return createNumOp<NumCellsOp>(loc, func, args);

    // ********************************************************************
    // Elementwise unary
    // ********************************************************************

    // --------------------------------------------------------------------
    // Arithmetic/general math
    // --------------------------------------------------------------------

    if(func == "abs")
        return createEwUnaryOp<EwAbsOp>(loc, func, args);
    if(func == "sign")
        return createEwUnaryOp<EwSignOp>(loc, func, args);
    if(func == "exp")
        return createEwUnaryOp<EwExpOp>(loc, func, args);
    if(func == "ln")
        return createEwUnaryOp<EwLnOp>(loc, func, args);
    if(func == "sqrt")
        return createEwUnaryOp<EwSqrtOp>(loc, func, args);

    // --------------------------------------------------------------------
    // Rounding
    // --------------------------------------------------------------------

    if(func == "round")
        return createEwUnaryOp<EwRoundOp>(loc, func, args);
    if(func == "floor")
        return createEwUnaryOp<EwFloorOp>(loc, func, args);
    if(func == "ceil")
        return createEwUnaryOp<EwCeilOp>(loc, func, args);

    // --------------------------------------------------------------------
    // Trigonometric
    // --------------------------------------------------------------------

    if(func == "sin")
        return createEwUnaryOp<EwSinOp>(loc, func, args);
    if(func == "cos")
        return createEwUnaryOp<EwCosOp>(loc, func, args);
    if(func == "tan")
        return createEwUnaryOp<EwTanOp>(loc, func, args);
    if(func == "sinh")
        return createEwUnaryOp<EwSinhOp>(loc, func, args);
    if(func == "cosh")
        return createEwUnaryOp<EwCoshOp>(loc, func, args);
    if(func == "tanh")
        return createEwUnaryOp<EwTanhOp>(loc, func, args);
    if(func == "asin")
        return createEwUnaryOp<EwAsinOp>(loc, func, args);
    if(func == "acos")
        return createEwUnaryOp<EwAcosOp>(loc, func, args);
    if(func == "atan")
        return createEwUnaryOp<EwAtanOp>(loc, func, args);

    // ********************************************************************
    // Elementwise binary
    // ********************************************************************

    // --------------------------------------------------------------------
    // Arithmetic
    // --------------------------------------------------------------------

    if(func == "pow")
        return createEwBinaryOp<EwPowOp>(loc, func, args);
    if(func == "log")
        return createEwBinaryOp<EwLogOp>(loc, func, args);

    // --------------------------------------------------------------------
    // Min/max
    // --------------------------------------------------------------------

    if(func == "min")
        return createEwBinaryOp<EwMinOp>(loc, func, args);
    if(func == "max")
        return createEwBinaryOp<EwMaxOp>(loc, func, args);

    // ********************************************************************
    // Aggregation and statistical
    // ********************************************************************

    // --------------------------------------------------------------------
    // Full aggregation, row/column-wise aggregation, grouped aggregation
    // --------------------------------------------------------------------
    // These four kinds of aggregation all have the same built-in function
    // names and are distinguished by their arguments.

    if(func == "sum")
        return createAnyAggOp<AllAggSumOp, RowAggSumOp, ColAggSumOp, GrpAggSumOp>(loc, func, args);
    if(func == "aggMin") // otherwise name clash with elementwise functions (cannot be resolved by types)
        return createAnyAggOp<AllAggMinOp, RowAggMinOp, ColAggMinOp, GrpAggMinOp>(loc, func, args);
    if(func == "aggMax") // otherwise name clash with elementwise functions (cannot be resolved by types)
        return createAnyAggOp<AllAggMaxOp, RowAggMaxOp, ColAggMaxOp, GrpAggMaxOp>(loc, func, args);
    if(func == "mean")
        return createAnyAggOp<AllAggMeanOp, RowAggMeanOp, ColAggMeanOp, GrpAggMeanOp>(loc, func, args);
    if(func == "var")
        return createAnyAggOp<AllAggVarOp, RowAggVarOp, ColAggVarOp, GrpAggVarOp>(loc, func, args);
    if(func == "stddev")
        return createAnyAggOp<AllAggStddevOp, RowAggStddevOp, ColAggStddevOp, GrpAggStddevOp>(loc, func, args);
    if(func == "idxMin")
        return createRowOrColAggOp<RowAggIdxMinOp, ColAggIdxMinOp>(loc, func, args);
    if(func == "idxMax")
        return createRowOrColAggOp<RowAggIdxMaxOp, ColAggIdxMaxOp>(loc, func, args);
    if(func == "count")
        return createGrpAggOp<GrpAggCountOp>(loc, func, args);

    // --------------------------------------------------------------------
    // Cumulative aggregation
    // --------------------------------------------------------------------

    if(func == "cumSum")
        return createCumAggOp<CumAggSumOp>(loc, func, args);
    if(func == "cumProd")
        return createCumAggOp<CumAggProdOp>(loc, func, args);
    if(func == "cumMin")
        return createCumAggOp<CumAggMinOp>(loc, func, args);
    if(func == "cumMax")
        return createCumAggOp<CumAggMaxOp>(loc, func, args);

    // --------------------------------------------------------------------
    // Statistical for column matrices
    // --------------------------------------------------------------------

    // TODO Add built-in functions for those.

    // ********************************************************************
    // Reorganization
    // ********************************************************************

    if(func == "reshape") {
        checkNumArgsExact(func, numArgs, 3);
        mlir::Value arg = args[0];
        mlir::Value numRows = utils.castSizeIf(args[1]);
        mlir::Value numCols = utils.castSizeIf(args[2]);
        return static_cast<mlir::Value>(builder.create<ReshapeOp>(
                loc, arg.getType(), arg, numRows, numCols
        ));
    }
    if(func == "transpose" || func == "t") {
        checkNumArgsExact(func, numArgs, 1);
        return static_cast<mlir::Value>(builder.create<TransposeOp>(
                loc, args[0]
        ));
    }
    if(func == "cbind")
        return createBindOp<ColBindOp>(loc, func, args);
    if(func == "rbind")
        return createBindOp<RowBindOp>(loc, func, args);
    if(func == "reverse")
        return createSameTypeUnaryOp<ReverseOp>(loc, func, args);
    if(func == "order") {
        checkNumArgsMin(func, numArgs, 4);
        checkNumArgsEven(func, numArgs);
        mlir::Value arg = args[0];
        std::vector<mlir::Value> colIdxs;
        std::vector<mlir::Value> ascs;
        bool returnIdxs = false; // TODO Don't hardcode this.
        const size_t numCols = (numArgs - 2) / 2;
        for(size_t i = 0; i < numCols; i++) {
            colIdxs.push_back(utils.castSizeIf(args[1 + i]));
            ascs.push_back(utils.castBoolIf(args[1 + numCols + i]));
        }
        return static_cast<mlir::Value>(builder.create<OrderOp>(
                loc, args[0].getType(), arg, colIdxs, ascs, returnIdxs
        ));
    }

    // ********************************************************************
    // Matrix decompositions & co
    // ********************************************************************

    // TODO Add built-in functions for those.

    // ********************************************************************
    // Deep neural network
    // ********************************************************************

    // TODO Add built-in functions for those.

    // ********************************************************************
    // Other matrix operations
    // ********************************************************************

    if(func == "diagVector")
        return createSameTypeUnaryOp<DiagVectorOp>(loc, func, args);
    if(func == "lowerTri")
        return createTriOp(loc, func, args, false);
    if(func == "upperTri")
        return createTriOp(loc, func, args, true);
    if(func == "solve") {
        checkNumArgsExact(func, numArgs, 2);
        mlir::Value a = args[0];
        mlir::Value b = args[1];
        return static_cast<mlir::Value>(builder.create<SolveOp>(
                loc, b.getType(), a, b
        ));
    }
    if(func == "replace") {
        checkNumArgsExact(func, numArgs, 3);
        mlir::Value arg = args[0];
        mlir::Value pattern = args[1];
        mlir::Value replacement = args[2];
        return static_cast<mlir::Value>(builder.create<ReplaceOp>(
                loc, arg.getType(), arg, pattern, replacement
        ));
    }
    if(func == "ctable") {
        checkNumArgsExact(func, numArgs, 5);
        mlir::Value lhs = args[0];
        mlir::Value rhs = args[1];
        mlir::Value weights = args[2];
        mlir::Value outHeight = utils.castSizeIf(args[3]);
        mlir::Value outWidth = utils.castSizeIf(args[4]);
        return static_cast<mlir::Value>(builder.create<CTableOp>(
                loc, lhs.getType(), lhs, rhs, weights, outHeight, outWidth
        ));
    }

    // ********************************************************************
    // Extended relational algebra
    // ********************************************************************
    
    // ----------------------------------------------------------------------------
    // Entire SQL query
    // ----------------------------------------------------------------------------

    if(func == "sql") {
        checkNumArgsExact(func, numArgs, 1);
        if(auto co = args[0].getDefiningOp<mlir::daphne::ConstantOp>()) {
            mlir::Attribute attr = co.value();
            if(attr.isa<mlir::StringAttr>()) {
                // TODO How to know the column types, or how to not need to
                // know them here? For now, we just leave them blank here.
                std::vector<mlir::Type> colTypes;
                co.erase();
                return static_cast<mlir::Value>(builder.create<SqlOp>(
                        loc,
                        FrameType::get(builder.getContext(), colTypes),
                        attr.dyn_cast<mlir::StringAttr>()
                ));
            }
        }
        throw std::runtime_error("SqlOp requires a SQL query as a constant string");
    }

    // --------------------------------------------------------------------
    // Set operations
    // --------------------------------------------------------------------

    if(func == "intersect")
        return createSetOp<IntersectOp>(loc, func, args);
    if(func == "merge")
        return createSetOp<IntersectOp>(loc, func, args);
    if(func == "except")
        return createSetOp<IntersectOp>(loc, func, args);

    // --------------------------------------------------------------------
    // Cartesian product and joins
    // --------------------------------------------------------------------

    if(func == "cartesian") {
        checkNumArgsMin(func, numArgs, 2);
        std::vector<mlir::Type> colTypes;
        for(auto arg : args)
            for(mlir::Type t : arg.getType().dyn_cast<FrameType>().getColumnTypes())
                colTypes.push_back(t);
        return static_cast<mlir::Value>(builder.create<CartesianOp>(
                loc, FrameType::get(builder.getContext(), colTypes), args
        ));
    }
    if(func == "innerJoin")
        return createJoinOp<InnerJoinOp>(loc, func, args);
    if(func == "fullOuterJoin")
        return createJoinOp<FullOuterJoinOp>(loc, func, args);
    if(func == "leftOuterJoin")
        return createJoinOp<LeftOuterJoinOp>(loc, func, args);
    if(func == "antiJoin")
        return createJoinOp<AntiJoinOp>(loc, func, args);
    if(func == "semiJoin")
        return createJoinOp<SemiJoinOp>(loc, func, args);

    // ********************************************************************
    // Conversions, casts, and copying
    // ********************************************************************

    if(func == "copy")
        return createSameTypeUnaryOp<CopyOp>(loc, func, args);

    // ********************************************************************
    // Input/output
    // ********************************************************************

    if(func == "print") {
        checkNumArgsExact(func, numArgs, 1);
        return builder.create<PrintOp>(
                loc, args[0]
        );
    }
    // TODO read/write
    
    // ********************************************************************
    // Data preprocessing
    // ********************************************************************
    
    if(func == "oneHot") {
        checkNumArgsExact(func, numArgs, 2);
        mlir::Value arg = args[0];
        mlir::Value info = args[1];
        return static_cast<mlir::Value>(builder.create<OneHotOp>(
                loc, arg.getType(), arg, info
        ));
    }

    // ********************************************************************

    throw std::runtime_error("unknown built-in function: '" + func + "'");
}