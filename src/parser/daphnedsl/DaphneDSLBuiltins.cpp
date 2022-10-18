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

#include <compiler/CompilerUtils.h>
#include <ir/daphneir/Daphne.h>
#include <parser/daphnedsl/DaphneDSLBuiltins.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/io/FileMetaData.h>

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
    return utils.retValWithInferedType(builder.create<EwUnaryOp>(
            loc, utils.unknownType, args[0]
    ));
}

template<class EwBinaryOp>
mlir::Value DaphneDSLBuiltins::createEwBinaryOp(mlir::Location loc, const std::string & func, const std::vector<mlir::Value> & args) {
    checkNumArgsExact(func, args.size(), 2);
    return utils.retValWithInferedType(builder.create<EwBinaryOp>(
            loc, args[0], args[1]
    ));
}

template<class RowAggOp, class ColAggOp>
mlir::Value DaphneDSLBuiltins::createRowOrColAggOp(mlir::Location loc, const std::string & func, const std::vector<mlir::Value> & args) {
    checkNumArgsExact(func, args.size(), 2);
    if(auto co = args[1].getDefiningOp<mlir::daphne::ConstantOp>()) {
        llvm::APInt axis = co.value().dyn_cast<mlir::IntegerAttr>().getValue();
        if(axis == 0)
            return utils.retValWithInferedType(
                    builder.create<RowAggOp>(
                            loc, utils.unknownType, args[0]
                    )
            );
        else if(axis == 1)
            return utils.retValWithInferedType(
                    builder.create<ColAggOp>(
                            loc, utils.unknownType, args[0]
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
        return utils.retValWithInferedType(builder.create<AllAggOp>(loc, utils.unknownType, args[0]));
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
    return utils.retValWithInferedType(builder.create<BindOp>(
            loc, utils.unknownType, args[0], args[1]
    ));
}

template<class TheOp>
mlir::Value DaphneDSLBuiltins::createSameTypeUnaryOp(mlir::Location loc, const std::string & func, const std::vector<mlir::Value> & args) {
    checkNumArgsExact(func, args.size(), 1);
    return utils.retValWithInferedType(builder.create<TheOp>(
            loc, utils.unknownType, args[0]
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

mlir::Value DaphneDSLBuiltins::createAffineFwdOp(mlir::Location loc, const std::string& func, const std::vector<mlir::Value>& args) {
    const size_t numArgs = args.size();
    checkNumArgsExact(func, numArgs, 3);

    mlir::Value input_data = args[0];
    mlir::Value weights_data = args[1];
    mlir::Value bias_data = args[2];

    return static_cast<mlir::Value>(builder.create<mlir::daphne::AffineForwardOp>(loc, input_data.getType(), input_data,
        weights_data, bias_data));
}

mlir::Value DaphneDSLBuiltins::createBatchNorm2dTestFwdOp(mlir::Location loc, const std::string &func,
        const std::vector<mlir::Value> &args) {
    const size_t numArgs = args.size();
    checkNumArgsExact(func, numArgs, 6);

    mlir::Value input_data = args[0];
    mlir::Value gamma = args[1];
    mlir::Value beta = args[2];

    mlir::Value ema_mean = args[3];
    mlir::Value ema_var = args[4];
    mlir::Value eps = args[5];

    return  static_cast<mlir::Value>(builder.create<mlir::daphne::BatchNorm2DTestForwardOp>(loc, input_data.getType(),
            input_data, gamma, beta, ema_mean, ema_var, eps));
}

mlir::ResultRange DaphneDSLBuiltins::createConv2dFwdOp(mlir::Location loc, const std::string& func, const std::vector<mlir::Value>&
        args) {
    const size_t numArgs = args.size();
    checkNumArgsBetween(func, numArgs, 12, 13);

    mlir::Value input_data = args[0];
    mlir::Value filter_data = args[1];
    mlir::Value num_images = utils.castSizeIf(args[2]);
    mlir::Value num_channels = utils.castSizeIf(args[3]);
    mlir::Value img_height = utils.castSizeIf(args[4]);
    mlir::Value img_width = utils.castSizeIf(args[5]);

    mlir::Value filter_h = utils.castSizeIf(args[6]);
    mlir::Value filter_w = utils.castSizeIf(args[7]);
    mlir::Value stride_h = utils.castSizeIf(args[8]);
    mlir::Value stride_w = utils.castSizeIf(args[9]);
    mlir::Value padding_h = utils.castSizeIf(args[10]);
    mlir::Value padding_w = utils.castSizeIf(args[11]);
    if (numArgs == 12) {
        return builder.create<mlir::daphne::Conv2DForwardOp>(loc, input_data.getType(), utils.sizeType, utils.sizeType,
                input_data, filter_data, filter_data, num_images, num_channels, img_height, img_width, filter_h, filter_w, stride_h,
                stride_w, padding_h, padding_w).getResults();
    }
    else {
        mlir::Value bias = args[12];
        return builder.create<mlir::daphne::Conv2DForwardOp>(loc, input_data.getType(), utils.sizeType, utils.sizeType,
                input_data, filter_data, bias, num_images, num_channels, img_height, img_width, filter_h, filter_w, stride_h,
                stride_w, padding_h, padding_w).getResults();
    }
}

template<class PoolOp>
mlir::ResultRange DaphneDSLBuiltins::createPoolFwdOp(mlir::Location loc, const std::string& func,
        const std::vector<mlir::Value>&    args) {
    const size_t numArgs = args.size();
    checkNumArgsExact(func, numArgs, 11);

    mlir::Value input_data = args[0];
    mlir::Value num_images = utils.castSizeIf(args[1]);
    mlir::Value num_channels = utils.castSizeIf(args[2]);
    mlir::Value img_height = utils.castSizeIf(args[3]);
    mlir::Value img_width = utils.castSizeIf(args[4]);
    mlir::Value pool_h = utils.castSizeIf(args[5]);
    mlir::Value pool_w = utils.castSizeIf(args[6]);
    mlir::Value stride_h = utils.castSizeIf(args[7]);
    mlir::Value stride_w = utils.castSizeIf(args[8]);
    mlir::Value padding_h = utils.castSizeIf(args[9]);
    mlir::Value padding_w = utils.castSizeIf(args[10]);

    return builder.create<PoolOp>(loc, input_data.getType(), utils.sizeType, utils.sizeType,
            input_data, num_images, num_channels, img_height, img_width, pool_h, pool_w, stride_h, stride_w, padding_h,
            padding_w).getResults();
}

// ****************************************************************************
// Other utilities
// ****************************************************************************

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
    if(func == "createFrame") {
        checkNumArgsMin(func, numArgs, 1);
        // Determine which arguments are column matrices and which are labels.
        std::vector<mlir::Type> colTypes;
        std::vector<mlir::Value> cols;
        std::vector<mlir::Value> labels;
        bool expectCol = true;
        for(auto arg : args) {
            mlir::Type t = arg.getType();
            auto mt = t.dyn_cast<MatrixType>();
            if(expectCol && mt) {
                colTypes.push_back(mt.getElementType());
                cols.push_back(arg);
            }
            else if(t.isa<mlir::daphne::StringType>()) {
                expectCol = false;
                labels.push_back(arg);
            }
            else
                throw std::runtime_error(
                        "arguments to frame() built-in function must be one or "
                        "more matrices optionally followed by equally many "
                        "strings"
                );
        }
        // Use default labels, if necessary.
        const size_t numCols = cols.size();
        const size_t numLabels = labels.size();
        if(!numLabels)
            for(size_t i = 0; i < numCols; i++) {
                const std::string dl(Frame::getDefaultLabel(i));
                labels.push_back(builder.create<ConstantOp>(
                        loc, dl
                ));
            }
        else if(numLabels != numCols)
            throw std::runtime_error(
                    "frame built-in function expects either no column labels "
                    "or as many labels as columns"
            );
        // Create CreateFrameOp.
        mlir::Type t = FrameType::get(builder.getContext(), colTypes);
        return static_cast<mlir::Value>(
                builder.create<CreateFrameOp>(loc, t, cols, labels)
        );
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
        mlir::Value range = args[0];
        mlir::Value size = utils.castSizeIf(args[1]);
        mlir::Value withReplacement = utils.castBoolIf(args[2]);
        mlir::Value seed = utils.castSeedIf(args[3]);
        return static_cast<mlir::Value>(
                builder.create<SampleOp>(
                        loc,
                        MatrixType::get(builder.getContext(), range.getType()),
                        range, size, withReplacement, seed
                )
        );
    }
    if(func == "seq") {
        checkNumArgsExact(func, numArgs, 3);
        mlir::Value from = args[0];
        mlir::Value to = args[1];
        mlir::Value inc= args[2];
        return utils.retValWithInferedType(
                builder.create<SeqOp>(
                        loc, utils.unknownType, from, to, inc
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
    if(func == "mod")
        return createEwBinaryOp<EwModOp>(loc, func, args);
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

    // --------------------------------------------------------------------
    // Strings
    // --------------------------------------------------------------------

    if(func == "concat") {
        checkNumArgsExact(func, numArgs, 2);
        return static_cast<mlir::Value>(builder.create<ConcatOp>(
                loc, StringType::get(builder.getContext()), args[0], args[1]
        ));
    }

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
        mlir::Value returnIdxs = args[numArgs - 1];
        const size_t numCols = (numArgs - 2) / 2;
        for(size_t i = 0; i < numCols; i++) {
            colIdxs.push_back(utils.castSizeIf(args[1 + i]));
            ascs.push_back(utils.castBoolIf(args[1 + numCols + i]));
        }
        mlir::Type retTy;
        if(auto co = returnIdxs.getDefiningOp<ConstantOp>()) {
            if(co.value().dyn_cast<mlir::BoolAttr>().getValue())
                retTy = utils.matrixOfSizeType;
            else
                retTy = args[0].getType();
        }
        else
            retTy = utils.unknownType;
        return static_cast<mlir::Value>(builder.create<OrderOp>(
                loc, retTy, arg, colIdxs, ascs, returnIdxs
        ));
    }

    // ********************************************************************
    // Matrix decompositions & co
    // ********************************************************************

    if( func == "eigen" ) {
        checkNumArgsExact(func, numArgs, 1);
        //TODO JIT-Engine invocation failed: Failed to materialize symbols
        return builder.create<EigenOp>(loc,
            args[0].getType(), args[0].getType(), args[0]).getResults();
    }

    // TODO Add built-in functions for those.

    // ********************************************************************
    // Deep neural network
    // ********************************************************************

    if (func == "affine") {
        return createAffineFwdOp(loc, func, args);
    }

    if(func == "avg_pool2d") {
        return createPoolFwdOp<AvgPoolForwardOp>(loc, func, args);
    }

    if(func == "batch_norm2d") {
        return createBatchNorm2dTestFwdOp(loc, func, args);
    }

    if (func == "biasAdd") {
        checkNumArgsExact(func, numArgs, 2);
        mlir::Value input_data = args[0];
        mlir::Value bias = args[1];
        return static_cast<mlir::Value>(builder.create<mlir::daphne::BiasAddForwardOp>(loc, input_data.getType(),
                input_data, bias));
    }

    if(func == "conv2d") {
        return createConv2dFwdOp(loc, func, args);
    }

    if(func == "max_pool2d") {
        return createPoolFwdOp<MaxPoolForwardOp>(loc, func, args);
    }

    if (func == "relu") {
        checkNumArgsExact(func, numArgs, 1);
        mlir::Value input_data = args[0];
        return static_cast<mlir::Value>(builder.create<mlir::daphne::ReluForwardOp>(loc, input_data.getType(), input_data));
    }

    if (func == "softmax") {
        checkNumArgsExact(func, numArgs, 1);
        mlir::Value input_data = args[0];
        return static_cast<mlir::Value>(builder.create<mlir::daphne::SoftmaxForwardOp>(loc, input_data.getType(), input_data));
    }

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
        return utils.retValWithInferedType(builder.create<SolveOp>(
                loc, utils.unknownType, a, b
        ));
    }
    if(func == "replace") {
        checkNumArgsExact(func, numArgs, 3);
        mlir::Value arg = args[0];
        mlir::Value pattern = args[1];
        mlir::Value replacement = args[2];
        return utils.retValWithInferedType(builder.create<ReplaceOp>(
                loc, utils.unknownType, arg, pattern, replacement
        ));
    }
    if(func == "ctable") {
        checkNumArgsExact(func, numArgs, 2);
        mlir::Value lhs = args[0];
        mlir::Value rhs = args[1];
        // TODO Support all parameters of this operation again.
//        mlir::Value weights = args[2];
//        mlir::Value outHeight = utils.castSizeIf(args[3]);
//        mlir::Value outWidth = utils.castSizeIf(args[4]);
        return utils.retValWithInferedType(builder.create<CTableOp>(
//                loc, utils.unknownType, lhs, rhs, weights, outHeight, outWidth
                loc, utils.unknownType, lhs, rhs
        ));
    }
    if(func == "syrk") {
        return createSameTypeUnaryOp<SyrkOp>(loc, func, args);
    }
    if(func == "gemv") {
        checkNumArgsExact(func, numArgs, 2);
        mlir::Value mat = args[0];
        mlir::Value vec = args[1];
        return utils.retValWithInferedType(builder.create<GemvOp>(
                loc, utils.unknownType, mat, vec
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
                // TODO Don't hardcode the column types. Since we cannot know
                // them at this point, we should enable some way to leave them
                // unknown.
                colTypes.push_back(builder.getF64Type());
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
    if(func == "registerView") {
        checkNumArgsExact(func, numArgs, 2);
        auto co = args[0].getDefiningOp<mlir::daphne::ConstantOp>();
        mlir::Attribute attr = co.value();
        mlir::Value view = args[1];
        if(attr.isa<mlir::StringAttr>()) {
            co.erase();
            return builder.create<RegisterViewOp>(
                    loc,
                    attr.dyn_cast<mlir::StringAttr>(),
                    view
            );
        }

        throw std::runtime_error(
                "registerView requires a view name as a constant string, and "
                "a frame that gets assigned to that name"
        );

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
        checkNumArgsExact(func, numArgs, 2);
        std::vector<mlir::Type> colTypes;
        for(auto arg : args)
            for(mlir::Type t : arg.getType().dyn_cast<FrameType>().getColumnTypes())
                colTypes.push_back(t);
        return static_cast<mlir::Value>(builder.create<CartesianOp>(
                loc, FrameType::get(builder.getContext(), colTypes), args[0], args[1]
        ));
    }
    if(func == "innerJoin"){
        checkNumArgsExact(func, numArgs, 4);
        std::vector<mlir::Type> colTypes;
        for(int i = 0; i < 2; i++)
            for(mlir::Type t : args[i].getType().dyn_cast<FrameType>().getColumnTypes())
                colTypes.push_back(t);
        return static_cast<mlir::Value>(builder.create<InnerJoinOp>(
                loc, FrameType::get(builder.getContext(), colTypes), args[0], args[1], args[2], args[3]
        ));
    }
    if(func == "fullOuterJoin")
        return createJoinOp<FullOuterJoinOp>(loc, func, args);
    if(func == "leftOuterJoin")
        return createJoinOp<LeftOuterJoinOp>(loc, func, args);
    if(func == "antiJoin")
        return createJoinOp<AntiJoinOp>(loc, func, args);
    if(func == "semiJoin") {
        // TODO Reconcile this with the other join ops, but we need it to work
        // quickly now.
        // return createJoinOp<SemiJoinOp>(loc, func, args);
        checkNumArgsExact(func, numArgs, 4);
        mlir::Value lhs = args[0];
        mlir::Value rhs = args[1];
        mlir::Value lhsOn = args[2];
        mlir::Value rhsOn = args[3];
        return builder.create<SemiJoinOp>(
                loc,
                FrameType::get(
                        builder.getContext(),
                        {utils.unknownType}
                ),
                utils.matrixOfSizeType,
                lhs, rhs, lhsOn, rhsOn
        ).getResults();
    }
    if(func == "groupJoin") {
        checkNumArgsExact(func, numArgs, 5);
        mlir::Value lhs = args[0];
        mlir::Value rhs = args[1];
        mlir::Value lhsOn = args[2];
        mlir::Value rhsOn = args[3];
        mlir::Value rhsAgg = args[4];
        return builder.create<GroupJoinOp>(
                loc,
                FrameType::get(
                        builder.getContext(),
                        {utils.unknownType, utils.unknownType}
                ),
                utils.matrixOfSizeType,
                lhs, rhs, lhsOn, rhsOn, rhsAgg
        ).getResults();
    }

    // ********************************************************************
    // Frame label manipulation
    // ********************************************************************

    if(func == "setColLabels") {
        checkNumArgsMin(func, numArgs, 2);
        std::vector<mlir::Value> labels;
        for(size_t i = 1; i < numArgs; i++)
            labels.push_back(args[i]);
        return static_cast<mlir::Value>(builder.create<SetColLabelsOp>(
                loc,
                args[0].getType().dyn_cast<FrameType>().withSameColumnTypes(),
                args[0],
                labels
        ));
    }
    if(func == "setColLabelsPrefix") {
        checkNumArgsExact(func, numArgs, 2);
        return static_cast<mlir::Value>(builder.create<SetColLabelsPrefixOp>(
                loc,
                args[0].getType().dyn_cast<FrameType>().withSameColumnTypes(),
                args[0],
                args[1]
        ));
    }

    // ********************************************************************
    // Conversions, casts, and copying
    // ********************************************************************

    if(func == "copy") {
        return createSameTypeUnaryOp<CopyOp>(loc, func, args);
    }
    if(func == "quantize") {
        checkNumArgsExact(func, args.size(), 3);
        mlir::Value arg = args[0];
        mlir::Value min = args[1];
        mlir::Value max = args[2];
        return static_cast<mlir::Value>(builder.create<QuantizeOp>(
                loc,
                utils.matrixOf(builder.getIntegerType(8, false)),
                arg, min, max
        ));
    }

    // ********************************************************************
    // Input/output
    // ********************************************************************

    // --------------------------------------------------------------------
    // High-level
    // --------------------------------------------------------------------

    if(func == "print") {
        checkNumArgsBetween(func, numArgs, 1, 3);
        mlir::Value arg = args[0];
        mlir::Value newline = (numArgs < 2)
                ? builder.create<ConstantOp>(loc, builder.getBoolAttr(true))
                : utils.castBoolIf(args[1]);
        mlir::Value err = (numArgs < 3)
                ? builder.create<ConstantOp>(loc, builder.getBoolAttr(false))
                : utils.castBoolIf(args[2]);
        return builder.create<PrintOp>(
                loc, arg, newline, err
        );
    }
    if(func == "readFrame" || func == "readMatrix") {
        checkNumArgsExact(func, numArgs, 1);

        mlir::Value filename = args[0];
        FileMetaData fmd = CompilerUtils::getFileMetaData(filename);

        mlir::Type resType;

        if(func == "readFrame") {
            std::vector<mlir::Type> cts;
            if(fmd.isSingleValueType)
                for(size_t i = 0; i < fmd.numCols; i++)
                    cts.push_back(utils.mlirTypeForCode(fmd.schema[0]));
            else
                for(ValueTypeCode vtc : fmd.schema)
                    cts.push_back(utils.mlirTypeForCode(vtc));

            std::vector<std::string> * labels;
            if(fmd.labels.empty())
                labels = nullptr;
            else
                labels = new std::vector<std::string>(fmd.labels);

            resType = mlir::daphne::FrameType::get(
                    // TODO Inserting #rows/#cols here could cause problems, if
                    // the frame is involved in any SCF ops (if/while/for).
                    builder.getContext(), cts, fmd.numRows, fmd.numCols, labels
            );
        }
        else // func == "read.matrix"
            // If an individual value type was specified per column
            // (fmd.isSingleValueType == false), then this silently uses the
            // type of the first column.
            // TODO: add sparsity information here already (if present), currently not possible as many other ops
            //  just take input types as output types, which is incorrect for sparsity
            resType = utils.matrixOf(utils.mlirTypeForCode(fmd.schema[0]));

        return static_cast<mlir::Value>(builder.create<ReadOp>(
                loc, resType, filename
        ));
    }
    if(func == "writeFrame" || func == "writeMatrix" || func == "write") {
        // Note that the type of arg already indicates if it is a frame or a
        // matrix.
        checkNumArgsExact(func, numArgs, 2);
        mlir::Value arg = args[0];
        mlir::Value filename = args[1];
        return builder.create<WriteOp>(loc, arg, filename);
    }
    // --------------------------------------------------------------------
    // Low-level
    // --------------------------------------------------------------------

    if(func == "openFile") {
        checkNumArgsExact(func, numArgs, 1);
        mlir::Value filename = args[0];
        return static_cast<mlir::Value>(builder.create<OpenFileOp>(
                loc, FileType::get(builder.getContext()), filename
        ));
    }
    if(func == "openDevice") {
        checkNumArgsExact(func, numArgs, 1);
        mlir::Value device = args[0];
        return static_cast<mlir::Value>(builder.create<OpenDeviceOp>(
                loc, TargetType::get(builder.getContext()), device
        ));
    }
    if(func == "openFileOnTarget") {
        checkNumArgsExact(func, numArgs, 2);
        mlir::Value target = args[0];
        mlir::Value filename = args[1];
        return static_cast<mlir::Value>(builder.create<OpenFileOnTargetOp>(
                loc, DescriptorType::get(builder.getContext()), target, filename
        ));
    }
    if(func == "close") {
        checkNumArgsExact(func, numArgs, 1);
        mlir::Value fileOrTarget = args[0];
        return builder.create<CloseOp>(
                loc, fileOrTarget
        );
    }
    if(func == "readCsv") {
        checkNumArgsExact(func, numArgs, 4);
        mlir::Value fileOrDescriptor = args[0];
        mlir::Value numRows = utils.castSizeIf(args[1]);
        mlir::Value numCols = utils.castSizeIf(args[2]);
        mlir::Value delim = args[3];

        // TODO Currently, this always assumes double as the value type. We
        // need to connect this to our FileMetaData mechanism, but for that, we
        // require the file name, which is not known here in the current design.
        return static_cast<mlir::Value>(builder.create<ReadCsvOp>(
                loc, utils.matrixOf(builder.getF64Type()),
                fileOrDescriptor, numRows, numCols, delim
        ));
    }

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
    // Measurements
    // ********************************************************************

    if(func == "now") {
        checkNumArgsExact(func, numArgs, 0);
        return static_cast<mlir::Value>(builder.create<NowOp>(
                loc, builder.getIntegerType(64, true)
        ));
    }

    // ********************************************************************

    throw std::runtime_error("unknown built-in function: '" + func + "'");
}
