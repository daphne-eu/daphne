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
#include <compiler/utils/TypePrinting.h>
#include <ir/daphneir/Daphne.h>

#include <spdlog/spdlog.h>

#include <string>
#include <vector>
#include <stdexcept>

namespace mlir::daphne {
#include <ir/daphneir/DaphneInferTypesOpInterface.cpp.inc>
}

#include <compiler/inference/TypeInferenceUtils.h>

using namespace mlir;
using namespace mlir::OpTrait;

// ****************************************************************************
// General utility functions
// ****************************************************************************

Type getFrameColumnTypeByLabel(Operation *op, daphne::FrameType ft, Value labelVal) {
    auto labelStr = CompilerUtils::constantOrThrow<std::string>(labelVal,
            "the specified label must be a constant of string type");

    std::vector<std::string> * labels = ft.getLabels();
    if (labels) {
        // The column labels are known, so we search for the specified
        // label.
        std::vector<Type> colTypes = ft.getColumnTypes();
        for (size_t i = 0; i < colTypes.size(); i++)
            if ((*labels)[i] == labelStr)
                // Found the label.
                return colTypes[i];
        // Did not find the label.
        throw ErrorHandler::compilerError(
            op->getLoc(), "InferTypesOpInterface.cpp:" + std::to_string(__LINE__),
            "the specified label was not found: '" + labelStr + "'");
    } else
        // The column labels are unknown, so we cannot tell what type
        // the column with the specified label has.
        return daphne::UnknownType::get(ft.getContext());
}

// ****************************************************************************
// Type inference interface implementations
// ****************************************************************************

std::vector<Type> daphne::CastOp::inferTypes() {
    Type argTy = getArg().getType();
    Type resTy = getRes().getType();
    auto mtArg = argTy.dyn_cast<daphne::MatrixType>();
    auto ftArg = argTy.dyn_cast<daphne::FrameType>();
    auto mtRes = resTy.dyn_cast<daphne::MatrixType>();

    // If the result type is a matrix with so far unknown value type, then we
    // infer the value type from the argument.
    if(mtRes && llvm::isa<daphne::UnknownType>(mtRes.getElementType())) {
        Type resVt;

        if(mtArg)
            // The argument is a matrix; we use its value type for the result.
            resVt = mtArg.getElementType();
        else if(ftArg) {
            // The argument is a frame, we use the value type of its only
            // column for the results; if the argument has more than one
            // column, we throw an exception.
            std::vector<Type> ctsArg = ftArg.getColumnTypes();
            if(ctsArg.size() == 1)
                resVt = ctsArg[0];
            else
                // TODO We could use the most general of the column types.
                throw ErrorHandler::compilerError(
                    getLoc(), "InferTypesOpInterface (daphne::CastOp::inferTypes)",
                    "currently CastOp cannot infer the value type of its "
                    "output matrix, if the input is a multi-column frame");
        }
        else
            // The argument is a scalar, we use its type for the value type
            // of the result.
            // TODO double-check if it is really a scalar
            resVt = argTy;
         
        return {daphne::MatrixType::get(getContext(), resVt)};
    }

    // Otherwise, we leave the result type as it is. We do not reset it to
    // unknown, since this could drop information that was explicitly
    // encoded in the CastOp.
    return {resTy};
}

std::vector<Type> daphne::ExtractColOp::inferTypes() {
    Type u = daphne::UnknownType::get(getContext());
    Type srcTy = getSource().getType();
    Type selTy = getSelectedCols().getType();
    Type resTy;

    if(auto srcMatTy = srcTy.dyn_cast<daphne::MatrixType>())
        // Extracting columns from a matrix retains the value type.
        resTy = srcMatTy.withSameElementType();
    else if(auto srcFrmTy = srcTy.dyn_cast<daphne::FrameType>()) {
        // Extracting columns from a frame may change the list of column value types (schema).
        std::vector<Type> resColTys;

        if(auto selStrTy = selTy.dyn_cast<daphne::StringType>()) {
            std::string label = CompilerUtils::constantOrThrow<std::string>(getSelectedCols());
            std::string delimiter = ".";
            const std::string frameName = label.substr(0, label.find(delimiter));
            const std::string colLabel = label.substr(label.find(delimiter) + delimiter.length(), label.length());
            if(colLabel.compare("*") == 0) {
                std::vector<std::string> labels = *srcFrmTy.getLabels();
                std::vector<mlir::Type> colTypes = srcFrmTy.getColumnTypes();
                for (size_t i = 0; i < labels.size(); i++) {
                    std::string labelFrameName = labels[i].substr(0, labels[i].find(delimiter));
                    if (labelFrameName.compare(frameName) == 0) {
                        resColTys.push_back(colTypes[i]);
                    }
                }
            } else {
                // Extracting a single column by its string label.
                resColTys = {getFrameColumnTypeByLabel(this->getOperation(), srcFrmTy, getSelectedCols())};
            }
        }
        else if(auto selMatTy = selTy.dyn_cast<daphne::MatrixType>()) {
            // Extracting columns by their positions (given as a column matrix).

            // We don't know the result column types, but if the shape of selectedCols
            // is known, we at least know the number of columns in the result
            // and set them all to unknown type.
            const ssize_t numColsSel = selMatTy.getNumCols();
            const ssize_t numRowsSel = selMatTy.getNumRows();
            if(numColsSel != -1 && numColsSel != 1)
                throw ErrorHandler::compilerError(
                    getLoc(), "InferTypesOpInterface (daphne::ExtractColOp::inferTypes)",
                        "ExtractColOp type inference: selectedCols must have "
                        "exactly 1 column, but found " + std::to_string(numColsSel)
                );
            if(numRowsSel != -1)
                for(ssize_t i = 0; i < numRowsSel; i++)
                    resColTys.push_back(u);

            // TODO Use the concrete column positions whenever they are known, e.g.,
            // if selectedCols is defined by a MatrixConstantOp (matrix literal),
            // FillOp (with known scalar value), SeqOp, ...

            // TODO If all columns of the input frame have the same type, we know
            // the output frame's column types if we know the shape of selectedCols.
        }
        else
            throw ErrorHandler::compilerError(
                getLoc(),
                "InferTypesOpInterface (daphne::ExtractColOp::inferTypes)",
                "ExtractColOp type inference: selectedCols must be a string or "
                "a matrix");

        resTy = daphne::FrameType::get(getContext(), resColTys);
    }
    else
        resTy = u;

    return {resTy};
}

std::vector<Type> daphne::FilterColOp::inferTypes() {
    if (auto mt = getSource().getType().dyn_cast<daphne::MatrixType>())
        return {mt.withSameElementType()};
    else
        // TODO See #484.
        throw ErrorHandler::compilerError(
            getLoc(), "InferTypesOpInterface (daphne::FilterColOp::inferTypes)",
            "currently, FilterColOp can only infer its type for matrix inputs");
}

std::vector<Type> daphne::CreateFrameOp::inferTypes() {
    std::vector<Type> colTypes;
    for(Value col : getCols())
        colTypes.push_back(col.getType().dyn_cast<daphne::MatrixType>().getElementType());
    return {daphne::FrameType::get(getContext(), colTypes)};
}

std::vector<Type> daphne::RandMatrixOp::inferTypes() {
    auto elTy = getMin().getType();
    if(elTy == UnknownType::get(getContext())) {
        elTy = getMax().getType();
    }
    else {
        assert((getMax().getType() == UnknownType::get(getContext()) || elTy == getMax().getType())
            && "Min and max need to have the same type");
    }
    return {daphne::MatrixType::get(getContext(), elTy)};
}

std::vector<Type> daphne::EigenOp::inferTypes() {
    auto evMatType = getArg().getType().dyn_cast<daphne::MatrixType>();
    return  {evMatType.withSameElementType(), evMatType};
}

std::vector<Type> daphne::GroupJoinOp::inferTypes() {
    auto lhsFt = getLhs().getType().dyn_cast<daphne::FrameType>();
    auto rhsFt = getRhs().getType().dyn_cast<daphne::FrameType>();
    Type lhsOnType = getFrameColumnTypeByLabel(this->getOperation(), lhsFt, getLhsOn());
    Type rhsAggType = getFrameColumnTypeByLabel(this->getOperation(), rhsFt, getRhsAgg());

    MLIRContext * ctx = getContext();
    Builder builder(ctx);
    return {
        daphne::FrameType::get(ctx, {lhsOnType, rhsAggType}),
        daphne::MatrixType::get(ctx, builder.getIndexType())
    };
}

std::vector<Type> daphne::SemiJoinOp::inferTypes() {
    auto lhsFt = getLhs().getType().dyn_cast<daphne::FrameType>();
    Type lhsOnType = getFrameColumnTypeByLabel(this->getOperation(), lhsFt, getLhsOn());

    MLIRContext * ctx = getContext();
    Builder builder(ctx);
    return {
        daphne::FrameType::get(ctx, {lhsOnType}),
        daphne::MatrixType::get(ctx, builder.getIndexType())
    };
}

std::vector<Type> daphne::GroupOp::inferTypes() {
    MLIRContext * ctx = getContext();
    Builder builder(ctx);

    auto arg = getFrame().getType().dyn_cast<daphne::FrameType>();

    std::vector<Type> newColumnTypes;
    std::vector<Value> aggColValues;
    std::vector<std::string> aggFuncNames;

    for(Value t : getKeyCol()){
        //Key Types getting adopted for the new Frame
        std::string labelStr = CompilerUtils::constantOrThrow<std::string>(
            t, "the specified label must be a constant of string type"
        );
        std::string delimiter = ".";
        const std::string frameName = labelStr.substr(0, labelStr.find(delimiter));
        const std::string colLabel = labelStr.substr(labelStr.find(delimiter) + delimiter.length(), labelStr.length());
        if(labelStr == "*") {
            auto allTypes = arg.getColumnTypes();
            for (Type type: allTypes) {
                newColumnTypes.push_back(type);
            }
        } else if(colLabel.compare("*") == 0) {
            std::vector<std::string> labels = *arg.getLabels();
            std::vector<mlir::Type> colTypes = arg.getColumnTypes();
            for (size_t i = 0; i < labels.size(); i++) {
                std::string labelFrameName = labels[i].substr(0, labels[i].find(delimiter));
                if (labelFrameName.compare(frameName) == 0) {
                    newColumnTypes.push_back(colTypes[i]);
                }
            }
        } else {
            newColumnTypes.push_back(getFrameColumnTypeByLabel(this->getOperation(), arg, t));
        }
    }

    // Values get collected in an easier to use data structure
    for(Value t : getAggCol()){
        aggColValues.push_back(t);
    }
    // Function names get collected in an easier to use data structure
    for(Attribute t: getAggFuncs()){
        GroupEnum aggFuncValue = t.dyn_cast<GroupEnumAttr>().getValue();
        aggFuncNames.push_back(stringifyGroupEnum(aggFuncValue).str());
    }
    //New Types get computed
    for(size_t i = 0; i < aggFuncNames.size() && i < aggColValues.size(); i++){
        std::string groupAggFunction = aggFuncNames.at(i);
        if(groupAggFunction == "COUNT"){
            newColumnTypes.push_back(builder.getIntegerType(64, true));
        }else if(groupAggFunction == "AVG"){
            newColumnTypes.push_back(builder.getF64Type());
        }else{ //DEFAULT OPTION (The Type of the named column)
            Value t = aggColValues.at(i);
            newColumnTypes.push_back(getFrameColumnTypeByLabel(this->getOperation(), arg, t));
        }
    }
    return {daphne::FrameType::get(ctx, newColumnTypes)};
}

std::vector<Type> daphne::ExtractOp::inferTypes() {
    throw ErrorHandler::compilerError(
        getLoc(), "InferTypesOpInterface",
        "type inference not implemented for ExtractOp"); // TODO
}

std::vector<Type> daphne::OneHotOp::inferTypes() {
    throw ErrorHandler::compilerError(
        getLoc(), "InferTypesOpInterface",
        "type inference not implemented for OneHotOp"); // TODO
}

std::vector<Type> daphne::OrderOp::inferTypes() {
    // TODO Take into account if indexes or data shall be returned.
    Type srcType = getArg().getType();
    Type t;
    if(auto mt = srcType.dyn_cast<daphne::MatrixType>())
        t = mt.withSameElementType();
    else if(auto ft = srcType.dyn_cast<daphne::FrameType>())
        t = ft.withSameColumnTypes();
    return {t};
}


mlir::Type mlirTypeForCode(ValueTypeCode type, Builder builder) {
    switch(type) {
        case ValueTypeCode::SI8:  return builder.getIntegerType(8, true);
        case ValueTypeCode::SI32: return builder.getIntegerType(32, true);
        case ValueTypeCode::SI64: return builder.getIntegerType(64, true);
        case ValueTypeCode::UI8:  return builder.getIntegerType(8, false);
        case ValueTypeCode::UI32: return builder.getIntegerType(32, false);
        case ValueTypeCode::UI64: return builder.getIntegerType(64, false);
        case ValueTypeCode::F32: return builder.getF32Type();
        case ValueTypeCode::F64: return builder.getF64Type();
        default: throw std::runtime_error("mlirTypeForCode: unknown value type code");
    }
}

std::vector<Type> daphne::ReadOp::inferTypes() {

    auto p = CompilerUtils::isConstant<std::string>(getFileName());
    Builder builder(getContext());
    if (getRes().getType().dyn_cast<daphne::MatrixType>()) {
        // If an individual value type was specified per column
        // (fmd.isSingleValueType == false), then this silently uses the
        // type of the first column.
        // TODO: add sparsity information here already (if present), currently not possible as many other ops
        //  just take input types as output types, which is incorrect for sparsity
        if (p.first) {
            FileMetaData fmd = CompilerUtils::getFileMetaData(getFileName());
            mlir::Type valType = mlirTypeForCode(fmd.schema[0], builder);
            return {mlir::daphne::MatrixType::get(getContext(), valType)};
        } else {
            return {mlir::daphne::MatrixType::get(getContext(), daphne::UnknownType::get(getContext()))};
        }
    }
    else if (getRes().getType().dyn_cast<daphne::FrameType>()) {
        if (p.first) {
            FileMetaData fmd = CompilerUtils::getFileMetaData(getFileName());
            std::vector<mlir::Type> cts;
            if (fmd.isSingleValueType) {
                for (size_t i = 0; i < fmd.numCols; i++) {
                    cts.push_back(mlirTypeForCode(fmd.schema[0], builder));
                }
            } else {
                for (ValueTypeCode vtc : fmd.schema) {
                    cts.push_back(mlirTypeForCode(vtc, builder));
                }
            }
            return {mlir::daphne::FrameType::get(builder.getContext(), cts)};
        } else {
            return {mlir::daphne::FrameType::get(builder.getContext(), {daphne::UnknownType::get(getContext())})};
        }
    }
    return {daphne::UnknownType::get(getContext())};
}

std::vector<Type> daphne::SliceColOp::inferTypes() {
    Type u = daphne::UnknownType::get(getContext());
    Type srcTy = getSource().getType();
    Type resTy;

    if(auto srcMatTy = srcTy.dyn_cast<daphne::MatrixType>())
        // Slicing columns from a matrix retains the value type.
        resTy = srcMatTy.withSameElementType();
    else if(auto srcFrmTy = srcTy.dyn_cast<daphne::FrameType>()) {
        // Extracting columns from a frame may change the list of column value types (schema).
        auto loIn = CompilerUtils::isConstant<int64_t>(getLowerIncl());
        auto upEx = CompilerUtils::isConstant<int64_t>(getUpperExcl());
        if(loIn.first && upEx.first) {
            // Both the lower and upper bound are known.
            ssize_t loInPos = loIn.second;
            ssize_t upExPos = upEx.second;
            std::vector<Type> srcColTys = srcFrmTy.getColumnTypes();
            std::vector<Type> resColTys;

            // ToDo: remove this when dealing with the next ToDo below (just getting rid of a linter warning here)
            const auto srcNumCols = static_cast<ssize_t>(srcColTys.size());

            // TODO Don't duplicate these checks from shape inference.
            if(loInPos < 0 || loInPos >= srcNumCols)
                throw ErrorHandler::compilerError(
                    getLoc(), "InferTypesOpInterface",
                    "SliceColOp type inference: lowerIncl must be in [0, "
                    "numCols), "
                    "but is " +
                        std::to_string(loInPos) + " with " +
                        std::to_string(srcNumCols) + " cols");
            if(upExPos < 0 || upExPos > srcNumCols)
                throw ErrorHandler::compilerError(
                    getLoc(), "InferTypesOpInterface",
                    "SliceColOp type inference: upperExcl must be in [0, "
                    "numCols], "
                    "but is " +
                        std::to_string(upExPos) + " with " +
                        std::to_string(srcNumCols) + " cols");
            if(loInPos > upExPos)
                throw ErrorHandler::compilerError(
                    getLoc(), "InferTypesOpInterface",
                    "SliceColOp type inference: lowerIncl must not be greater "
                    "than upperExcl"
                    " (found " +
                        std::to_string(loInPos) + " and " +
                        std::to_string(upExPos) + ")");

            for(ssize_t pos = loInPos; pos < upExPos; pos++)
                resColTys.push_back(srcColTys[pos]);
                
            resTy = daphne::FrameType::get(getContext(), resColTys);
        }
        else
            // TODO The number of column types may not match the actual number of columns
            // in this case; actually, we should leave the column types blank, but this
            // cannot be represented at the moment.
            resTy = daphne::FrameType::get(getContext(), {u});
    }
    else
        resTy = u;

    return {resTy};
}

std::vector<Type> daphne::CondOp::inferTypes() {
    Type condTy = getCond().getType();
    if(llvm::isa<daphne::UnknownType>(condTy))
        return {daphne::UnknownType::get(getContext())};
    if(auto condMatTy = condTy.dyn_cast<daphne::MatrixType>()) {
        Type thenTy = getThenVal().getType();
        Type elseTy = getElseVal().getType();

        if(llvm::isa<daphne::FrameType>(thenTy) || llvm::isa<daphne::FrameType>(elseTy))
            throw ErrorHandler::compilerError(
                getLoc(), "InferTypesOpInterface",
                "CondOp does not support frames for the then-value or "
                "else-value if "
                "the condition is a matrix");

        Type thenValTy = CompilerUtils::getValueType(thenTy);
        Type elseValTy = CompilerUtils::getValueType(elseTy);

        if(thenValTy != elseValTy)
            throw ErrorHandler::compilerError(
                getLoc(), "InferTypesOpInterface",
                "the then/else-values of CondOp must have the same value type");

        return {daphne::MatrixType::get(getContext(), thenValTy)};
    }
    else if(auto condFrmTy = condTy.dyn_cast<daphne::FrameType>())
        throw ErrorHandler::compilerError(
            getLoc(), "InferTypesOpInterface",
            "CondOp does not support frames for the condition yet");
    else { // cond is a scalar // TODO check if it is really a scalar
        Type thenTy = getThenVal().getType();
        Type elseTy = getElseVal().getType();
        
        // Remove any properties of matrix/frame except for the value types,
        // such that they don't interfere with the type comparison below,
        // and since we don't want them in the inferred type.
        if(auto thenMatTy = thenTy.dyn_cast<daphne::MatrixType>())
            thenTy = thenMatTy.withSameElementType();
        else if(auto thenFrmTy = thenTy.dyn_cast<daphne::FrameType>())
            thenTy = thenFrmTy.withSameColumnTypes();
        if(auto elseMatTy = elseTy.dyn_cast<daphne::MatrixType>())
            elseTy = elseMatTy.withSameElementType();
        else if(auto elseFrmTy = elseTy.dyn_cast<daphne::FrameType>())
            elseTy = elseFrmTy.withSameColumnTypes();

        if(thenTy != elseTy) {
            throw ErrorHandler::compilerError(
                getLoc(), "InferTypesOpInterface (daphne::CondOp::inferTypes)",
                "the then/else-values of CondOp must have the same type if "
                "the condition is a scalar");
        }

        // It is important that all matrix/frame properties except for the
        // value have been removed.
        return {thenTy};
    }
}

std::vector<Type> daphne::RecodeOp::inferTypes() {
    // Intuition:
    // - The (data) result has the same data type as the argument.
    // - The (data) result has the value type si64.
    // - The (dict) result has the data type matrix.
    // - The (dict) result has the value type of the argument.
    //   - If the argument is a frame, all its columns must have the same
    //     value type (alternatively, one could take the most general one).

    MLIRContext * ctx = getContext();

    Type argTy = getArg().getType();
    Type si64 = IntegerType::get(ctx, 64, IntegerType::SignednessSemantics::Signed);
    Type u = daphne::UnknownType::get(ctx);

    Type resTy;
    Type dictValTy;
    if(auto argMatTy = llvm::dyn_cast<daphne::MatrixType>(argTy)) {
        resTy = daphne::MatrixType::get(ctx, si64);
        dictValTy = argMatTy.getElementType();
    }
    else if(auto argFrmTy = llvm::dyn_cast<daphne::FrameType>(argTy)) {
        std::vector<Type> argColTys = argFrmTy.getColumnTypes();
        if(argColTys.size() == 0) {
            resTy = daphne::FrameType::get(ctx, {});
            dictValTy = u;
        }
        else {
            std::vector<Type> resColTys(argColTys.size(), si64);
            resTy = daphne::FrameType::get(ctx, resColTys);
            dictValTy = nullptr;
            bool hasUnknownColTy = false;
            for(Type argColTy : argColTys) {
                if(llvm::isa<daphne::UnknownType>(argColTy))
                    hasUnknownColTy = true;
                else {
                    if(!dictValTy)
                        dictValTy = argColTy;
                    else if(argColTy != dictValTy)
                        throw ErrorHandler::compilerError(
                            getLoc(), "InferTypesOpInterface",
                            "when a frame is used as the argument to recode, "
                            "all its columns must have the same value type");
                }
            }
            if(!dictValTy || hasUnknownColTy)
                dictValTy = u;
        }
    }
    else if(llvm::isa<daphne::UnknownType>(argTy))
        resTy = u;
    else
        throw ErrorHandler::compilerError(
            getLoc(), "InferTypesOpInterface",
            "the argument to recode has an invalid type");

    Type dictTy = daphne::MatrixType::get(ctx, dictValTy);
    return {resTy, dictTy};
}

std::vector<Type> daphne::Conv2DForwardOp::inferTypes() {
    MLIRContext * ctx = getContext();
    Builder builder(ctx);
    auto restype = llvm::dyn_cast<daphne::MatrixType>(getInput().getType());
    auto restype2 = daphne::MatrixType::get(ctx, restype.getElementType());

    // output matrix of same type as input, height/width dimensions as size/index type
    return {restype2, builder.getIndexType(), builder.getIndexType()};
}

// ****************************************************************************
// Type inference function
// ****************************************************************************

std::vector<Type> daphne::tryInferType(Operation* op) {
    if(auto inferTypeOp = llvm::dyn_cast<daphne::InferTypes>(op))
        // If the operation implements the type inference interface,
        // we apply that.
        return inferTypeOp.inferTypes();
    else if(op->getNumResults() == 1) {
        // If the operation does not implement the type inference interface
        // and has exactly one result, we utilize its type inference traits.
        mlir::Type resTy;
        try {
            // Note that all our type inference traits assume that the operation
            // has exactly one result (which is the case for most DaphneIR ops).
            resTy = inferTypeByTraits<mlir::Operation>(op);
        } catch (std::runtime_error &e) {
            throw ErrorHandler::rethrowError("InferTypesOpInterface", e.what());
        }
        return {resTy};
    }
    else {
        // If the operation does not implement the type inference interface
        // and has zero or more than one results, we return unknowns.
        std::vector<Type> resTys;
        for(size_t i = 0; i < op->getNumResults(); i++)
            resTys.push_back(daphne::UnknownType::get(op->getContext()));
        return resTys;
    }
}

void daphne::setInferedTypes(Operation* op, bool partialInferenceAllowed) {
    // Try to infer the types of all results of this operation.
    std::vector<Type> types;
    try {
        types = daphne::tryInferType(op);
    }
    catch (std::runtime_error& re) {
        throw ErrorHandler::rethrowError(
            "InferTypesOpInterface.cpp:" + std::to_string(__LINE__), re.what());
    }
    catch (...) {
        throw ErrorHandler::rethrowError(
            "InferTypesOpInterface.cpp:" + std::to_string(__LINE__), "Unknown exception.");
    }
    const size_t numRes = op->getNumResults();
    if(types.size() != numRes)
        throw ErrorHandler::compilerError(
            op->getLoc(), "InferTypesOpInterface",
            "type inference for op " + op->getName().getStringRef().str() +
                " returned " + std::to_string(types.size()) +
                " types, but the op has " + std::to_string(numRes) +
                " results");
    // Set the inferred types on all results of this operation.
    for(size_t i = 0; i < numRes; i++) {
        if (llvm::isa<daphne::UnknownType>(types[i]) && !partialInferenceAllowed)
            // TODO As soon as the run-time can handle unknown
            // data/value types, we do not need to throw here anymore.
            throw ErrorHandler::compilerError(
                op->getLoc(), "InferTypesOpInterface",
                "type inference returned an unknown result type "
                "for some op, but partial inference is not allowed "
                "at this point: " +
                    op->getName().getStringRef().str());
        op->getResult(i).setType(types[i]);
    }
}
