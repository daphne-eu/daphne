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

Type getFrameColumnTypeByLabel(daphne::FrameType ft, Value labelVal) {
    std::string labelStr = CompilerUtils::constantOrThrow<std::string>(
            labelVal, "the specified label must be a constant of string type"
    );

    std::vector<std::string> * labels = ft.getLabels();
    if(labels) {
        // The column labels are known, so we search for the specified
        // label.
        std::vector<Type> colTypes = ft.getColumnTypes();
        for(size_t i = 0; i < colTypes.size(); i++)
            if((*labels)[i] == labelStr)
                // Found the label.
                return colTypes[i];
        // Did not find the label.
        throw std::runtime_error(
                "the specified label was not found: '" + labelStr + "'"
        );
    }
    else
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
    if(mtRes && mtRes.getElementType().isa<daphne::UnknownType>()) {
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
                throw std::runtime_error(
                        "currently CastOp cannot infer the value type of its "
                        "output matrix, if the input is a multi-column frame"
                );
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

        if(auto selStrTy = selTy.dyn_cast<daphne::StringType>())
            // Extracting a single column by its string label.
            resColTys = {getFrameColumnTypeByLabel(srcFrmTy, getSelectedCols())};
        else if(auto selMatTy = selTy.dyn_cast<daphne::MatrixType>()) {
            // Extracting columns by their positions (given as a column matrix).

            // We don't know the result column types, but if the shape of selectedCols
            // is known, we at least know the number of columns in the result
            // and set them all to unknown type.
            const ssize_t numColsSel = selMatTy.getNumCols();
            const ssize_t numRowsSel = selMatTy.getNumRows();
            if(numColsSel != -1 && numColsSel != 1)
                throw std::runtime_error(
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
            throw std::runtime_error(
                    "ExtractColOp type inference: selectedCols must be a string or a matrix"
            );
        
        resTy = daphne::FrameType::get(getContext(), resColTys);
    }
    else
        resTy = u;

    return {resTy};
}

std::vector<Type> daphne::FilterColOp::inferTypes() {
    if(auto mt = getSource().getType().dyn_cast<daphne::MatrixType>())
        return {mt.withSameElementType()};
    else
        // TODO See #484.
        throw std::runtime_error(
                "currently, FilterColOp can only infer its type for matrix inputs"
        );
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

std::vector<Type> daphne::GroupJoinOp::inferTypes() {
    daphne::FrameType lhsFt = getLhs().getType().dyn_cast<daphne::FrameType>();
    daphne::FrameType rhsFt = getRhs().getType().dyn_cast<daphne::FrameType>();
    Type lhsOnType = getFrameColumnTypeByLabel(lhsFt, getLhsOn());
    Type rhsAggType = getFrameColumnTypeByLabel(rhsFt, getRhsAgg());

    MLIRContext * ctx = getContext();
    Builder builder(ctx);
    return {
        daphne::FrameType::get(ctx, {lhsOnType, rhsAggType}),
        daphne::MatrixType::get(ctx, builder.getIndexType())
    };
}

std::vector<Type> daphne::SemiJoinOp::inferTypes() {
    daphne::FrameType lhsFt = getLhs().getType().dyn_cast<daphne::FrameType>();
    Type lhsOnType = getFrameColumnTypeByLabel(lhsFt, getLhsOn());

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

    daphne::FrameType arg = getFrame().getType().dyn_cast<daphne::FrameType>();

    std::vector<Type> newColumnTypes;
    std::vector<Value> aggColValues;
    std::vector<std::string> aggFuncNames;

    for(Value t : getKeyCol()){
        //Key Types getting adopted for the new Frame
        newColumnTypes.push_back(getFrameColumnTypeByLabel(arg, t));
    }

    // Values get collected in a easier to use Datastructure
    for(Value t : getAggCol()){
        aggColValues.push_back(t);
    }
    // Function names get collected in a easier to use Datastructure
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
            newColumnTypes.push_back(getFrameColumnTypeByLabel(arg, t));
        }
    }
    return {daphne::FrameType::get(ctx, newColumnTypes)};
}

std::vector<Type> daphne::ExtractOp::inferTypes() {
    throw std::runtime_error("type inference not implemented for ExtractOp"); // TODO
}

std::vector<Type> daphne::OneHotOp::inferTypes() {
    throw std::runtime_error("type inference not implemented for OneHotOp"); // TODO
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
            const ssize_t srcNumCols = srcColTys.size();

            // TODO Don't duplicate these checks from shape inference.
            if(loInPos < 0 || loInPos >= srcNumCols)
                throw std::runtime_error(
                    "SliceColOp type inference: lowerIncl must be in [0, numCols), "
                    "but is " + std::to_string(loInPos) +
                    " with " + std::to_string(srcNumCols) + " cols"
                );
            if(upExPos < 0 || upExPos > srcNumCols)
                throw std::runtime_error(
                    "SliceColOp type inference: upperExcl must be in [0, numCols], "
                    "but is " + std::to_string(upExPos) +
                    " with " + std::to_string(srcNumCols) + " cols"
                );
            if(loInPos > upExPos)
                throw std::runtime_error(
                    "SliceColOp type inference: lowerIncl must not be greater than upperExcl"
                    " (found " + std::to_string(loInPos) + " and " + std::to_string(upExPos) + ")"
                );

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
    if(condTy.isa<daphne::UnknownType>())
        return {daphne::UnknownType::get(getContext())};
    if(auto condMatTy = condTy.dyn_cast<daphne::MatrixType>()) {
        Type thenTy = getThenVal().getType();
        Type elseTy = getElseVal().getType();

        if(thenTy.isa<daphne::FrameType>() || elseTy.isa<daphne::FrameType>())
            throw std::runtime_error(
                 "CondOp does not support frames for the then-value or else-value if "
                 "the condition is a matrix"
            );

        Type thenValTy = CompilerUtils::getValueType(thenTy);
        Type elseValTy = CompilerUtils::getValueType(elseTy);

        if(thenValTy != elseValTy)
            throw std::runtime_error(
                    "the then/else-values of CondOp must have the same value type"
            );

        return {daphne::MatrixType::get(getContext(), thenValTy)};
    }
    else if(auto condFrmTy = condTy.dyn_cast<daphne::FrameType>())
        throw std::runtime_error("CondOp does not support frames for the condition yet");
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
            throw std::runtime_error(
                    "the then/else-values of CondOp must have the same type if "
                    "the condition is a scalar"
            );
        }

        // It is important that all matrix/frame properties except for the
        // value have been removed.
        return {thenTy};
    }
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
        
        mlir::Type resTy = inferTypeByTraits<mlir::Operation>(op);

        // Note that all our type inference traits assume that the operation
        // has exactly one result (which is the case for most DaphneIR ops).
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
        spdlog::error("Exception in {}:{}: \n{}",__FILE__, __LINE__, re.what());
        throw;
    }
    catch (...) {
        spdlog::error("Unknown exception in {}:{}",__FILE__, __LINE__);
        throw;
    }
    const size_t numRes = op->getNumResults();
    if(types.size() != numRes)
        throw std::runtime_error(
                "type inference for op " +
                op->getName().getStringRef().str() + " returned " +
                std::to_string(types.size()) + " types, but the op has " +
                std::to_string(numRes) + " results"
        );
    // Set the inferred types on all results of this operation.
    for(size_t i = 0; i < numRes; i++) {
        if (types[i].isa<daphne::UnknownType>() && !partialInferenceAllowed)
            // TODO As soon as the run-time can handle unknown
            // data/value types, we do not need to throw here anymore.
            throw std::runtime_error(
                    "type inference returned an unknown result type "
                    "for some op, but partial inference is not allowed "
                    "at this point: " + op->getName().getStringRef().str()
        );
        op->getResult(i).setType(types[i]);
    }
}