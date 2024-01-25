/*
 *  Copyright 2021 The DAPHNE Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <ir/daphneir/Daphne.h>

#include <mlir/IR/Operation.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Types.h>

#include <vector>

/**
 * @brief Returns an integer code representing how general a value type is.
 * 
 * This code can be used to determine which of two value types is more general.
 * The larger the code, the more general the value type.
 * 
 * @param t
 * @return 
 */
int generality(mlir::Type t);

/**
 * @brief Represents a data type for type inference.
 *
 * This is required since `mlir::Type` does not allow us to have a data type
 * without a value type.
 */
enum class DataTypeCode : uint8_t {
    // The greater the number, the more general the type.
    SCALAR, // least general
    MATRIX,
    FRAME,
    UNKNOWN // most general
};

/**
 * @brief Returns the most general value type in a list of value types.
 * 
 * @param vt A list of value types.
 * @return 
 */
mlir::Type mostGeneralVt(const std::vector<mlir::Type> & vt);

/**
 * @brief Returns the most general value type in a list of lists of value types.
 * 
 * @param vts A list of lists of value types.
 * @param num Optionally, only consider the first `num` lists of value types.
 * @return 
 */
mlir::Type mostGeneralVt(
        const std::vector<std::vector<mlir::Type>> & vts,
        size_t num = 0
);

/**
 * @brief Infers the value type assuming the type inference trait
 * `ValueTypeFromArgs`.
 * 
 * @param argDtc Information on the argument data types.
 * @param argVts Information on the argument value types.
 * @return The infered value type.
 */
std::vector<mlir::Type> inferValueTypeFromArgs(
        const std::vector<DataTypeCode> & argDtc,
        std::vector<std::vector<mlir::Type>> & argVts
);

/**
 * @brief Infers the type of the result of the given operation based on its
 * type inference traits.
 * 
 * @tparam O The type of the operation. For the inference in the compiler we
 * use `mlir::Operation`, but for the unit tests we use a mock class.
 * @param op
 * @return The infered type of the single result of the operation.
 */
template<class O>
mlir::Type inferTypeByTraits(O * op) {
    using namespace mlir;
    using namespace mlir::OpTrait;
    
    MLIRContext * ctx = op->getContext();
    Type u = daphne::UnknownType::get(ctx);
    
    Type resTy = u;

    // --------------------------------------------------------------------
    // Extract data types and value types
    // --------------------------------------------------------------------

    std::vector<DataTypeCode> argDtc;
    std::vector<std::vector<Type>> argVts;
    for(Type t : op->getOperandTypes()) {
        if(llvm::isa<daphne::UnknownType>(t)) {
            argDtc.push_back(DataTypeCode::UNKNOWN);
            argVts.push_back({u});
        }
        else if(auto ft = t.dyn_cast<daphne::FrameType>()) {
            argDtc.push_back(DataTypeCode::FRAME);
            argVts.push_back(ft.getColumnTypes());
        }
        else if(auto mt = t.dyn_cast<daphne::MatrixType>()) {
            argDtc.push_back(DataTypeCode::MATRIX);
            argVts.push_back({mt.getElementType()});
        }
        else { // TODO Check if this is really a supported scalar type!
            argDtc.push_back(DataTypeCode::SCALAR);
            argVts.push_back({t});
        }
    }

    // --------------------------------------------------------------------
    // Infer the data type
    // --------------------------------------------------------------------

    DataTypeCode resDtc = DataTypeCode::UNKNOWN;

    if(op->template hasTrait<DataTypeFromFirstArg>() || op->template hasTrait<TypeFromFirstArg>())
        resDtc = argDtc[0];
    else if(op->template hasTrait<DataTypeFromArgs>()) {
        resDtc = argDtc[0];
        for(size_t i = 1; i < argDtc.size(); i++)
            if(argDtc[i] > resDtc) // generality comparison
                resDtc = argDtc[i];
    }
    else if(op->template hasTrait<DataTypeSca>())
        resDtc = DataTypeCode::SCALAR;
    else if(op->template hasTrait<DataTypeMat>())
        resDtc = DataTypeCode::MATRIX;
    else if(op->template hasTrait<DataTypeFrm>())
        resDtc = DataTypeCode::FRAME;

    // --------------------------------------------------------------------
    // Infer the value type
    // --------------------------------------------------------------------

    // TODO What about the #cols, if the data type is a frame (see #421)?
    std::vector<Type> resVts = {u};

    if(op->template hasTrait<TypeFromFirstArg>())
        resVts = argVts[0];
    else if(op->template hasTrait<ValueTypeFromFirstArg>()) {
        if(resDtc == DataTypeCode::FRAME && argDtc[0] == DataTypeCode::MATRIX) {
            // We need to make sure that the value type of the input matrix is
            // repeated in the column value types of the output frame to match
            // the number of columns of the input matrix.
            const ssize_t numCols = op->getOperand(0)
                .getType()
                .template dyn_cast<daphne::MatrixType>()
                .getNumCols();
            if(numCols == -1)
                // The input's number of columns is unknown.
                resVts = {u}; // TODO How to properly represent such cases (see #421)?
            else
                // The input's number of columns is known.
                resVts = std::vector(numCols, argVts[0][0]);
        }
        else
            // Even if the first arg is a frame, its column types get collapsed
            // to the most general type later on.
            resVts = argVts[0];
    }
    // TODO Reduce the code duplication. Merge the traits ValueTypeFromFirstArg and
    // ValueTypeFromThirdArg into one parametric trait ValueTypeFromArg<N>, see #487.
    else if(op->template hasTrait<ValueTypeFromThirdArg>()) {
        if(resDtc == DataTypeCode::FRAME && argDtc[2] == DataTypeCode::MATRIX) {
            // We need to make sure that the value type of the input matrix is
            // repeated in the column value types of the output frame to match
            // the number of columns of the input matrix.
            const ssize_t numCols = op->getOperand(2)
                .getType()
                .template dyn_cast<daphne::MatrixType>()
                .getNumCols();
            if(numCols == -1)
                // The input's number of columns is unknown.
                resVts = {u}; // TODO How to properly represent such cases (see #421)?
            else
                // The input's number of columns is known.
                resVts = std::vector(numCols, argVts[2][0]);
        }
        else
            // Even if the third arg is a frame, its column types get collapsed
            // to the most general type later on.
            resVts = argVts[2];
    }
    else if(op->template hasTrait<ValueTypeFromArgs>())
        resVts = inferValueTypeFromArgs(argDtc, argVts);
    else if(op->template hasTrait<ValueTypeFromArgsFP>()) {
        // Get the most general value types...
        resVts = inferValueTypeFromArgs(argDtc, argVts);

        // ...and replace them by the most general floating-point type where
        // necessary.
        for(size_t i = 0; i < resVts.size(); i++)
            if(!llvm::isa<FloatType>(resVts[i]) && !llvm::isa<daphne::UnknownType>(resVts[i]))
                resVts[i] = FloatType::getF64(ctx);
    }
    else if(op->template hasTrait<ValueTypeFromArgsInt>()) {
        // Get the most general value types...
        resVts = inferValueTypeFromArgs(argDtc, argVts);

        // ...and replace them by the most general integer type where
        // necessary.
        for(size_t i = 0; i < resVts.size(); i++)
            if(!llvm::isa<IntegerType>(resVts[i]) && !llvm::isa<daphne::UnknownType>(resVts[i]))
                resVts[i] = IntegerType::get(
                        ctx, 64, IntegerType::SignednessSemantics::Unsigned
                );
    }
    else if(op->template hasTrait<ValueTypesConcat>()) {
        const size_t numArgsConsider = 2;
        if(argVts.size() < numArgsConsider)
            throw std::runtime_error(
                    "type inference trait ValueTypesConcat requires at least "
                    "two arguments"
            );

        switch(resDtc) {
            case DataTypeCode::FRAME:
                resVts = {};
                for(size_t i = 0; i < numArgsConsider; i++) {
                    bool abort = false;
                    switch(argDtc[i]) {
                        case DataTypeCode::FRAME:
                            // Append this input frame's column types to the
                            // result's column types.
                            for(size_t k = 0; k < argVts[i].size(); k++)
                                resVts.push_back(argVts[i][k]);
                            break;
                        case DataTypeCode::MATRIX: {
                            const ssize_t numCols = op->getOperand(i)
                                .getType()
                                .template dyn_cast<daphne::MatrixType>()
                                .getNumCols();
                            if(numCols == -1) {
                                // The number of columns of this input matrix
                                // is unknown, so it is unclear how often its
                                // value type needs to be appended to the
                                // result column types.
                                resVts = {u}; // TODO How to best represent this case (see #421)?
                                abort = true;
                            }
                            else
                                // The number of columns of this input matrix
                                // is known, so we append its value type to the
                                // result column types that number of times.
                                for(ssize_t k = 0; k < numCols; k++)
                                    resVts.push_back(argVts[i][0]);
                            break;
                        }
                        case DataTypeCode::SCALAR:
                            // Append the value type of this input scalar to
                            // the result column types.
                            resVts.push_back(argVts[i][0]);
                            break;
                        case DataTypeCode::UNKNOWN:
                            // It is unclear how this input contributes to
                            // the result's column types.
                            resVts = {u}; // TODO How to best represent this case (see #421)?
                            abort = true;
                            break;
                    }
                    if(abort)
                        break;
                }
                break;
            case DataTypeCode::MATRIX: // fall-through intended
            case DataTypeCode::SCALAR:
                resVts = {mostGeneralVt(argVts, numArgsConsider)};
                break;
            case DataTypeCode::UNKNOWN:
                // nothing to do
                break;
        }
    }
    else if(op->template hasTrait<ValueTypeSI64>())
        resVts = {IntegerType::get(ctx, 64, IntegerType::SignednessSemantics::Signed)};
    else if(op->template hasTrait<ValueTypeSize>())
        resVts = {IndexType::get(ctx)};
    else if(op->template hasTrait<ValueTypeStr>())
        resVts = {daphne::StringType::get(ctx)};

    // --------------------------------------------------------------------
    // Create the result type
    // --------------------------------------------------------------------

    // It is important to recreate matrix and frame types (not reuse those from
    // the inputs) to get rid of any additional properties (shape, etc.).
    switch(resDtc) {
        case DataTypeCode::UNKNOWN:
            resTy = u;
            break;
        case DataTypeCode::SCALAR:
            resTy = mostGeneralVt(resVts);
            break;
        case DataTypeCode::MATRIX:
            resTy = daphne::MatrixType::get(ctx, mostGeneralVt(resVts));
            break;
        case DataTypeCode::FRAME: {
            resTy = daphne::FrameType::get(ctx, resVts);
            break;
        }
    }

    // Note that all our type inference traits assume that the operation has
    // exactly one result (which is the case for most DaphneIR ops).
    return resTy;
}
