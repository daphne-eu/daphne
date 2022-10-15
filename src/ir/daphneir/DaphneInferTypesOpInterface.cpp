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

Type getFrameColumnTypeByLabel(daphne::FrameType ft, Value label) {
    // TODO Use getConstantString from DaphneInferFrameLabelsOpInterface.cpp.
    if(auto co = llvm::dyn_cast<daphne::ConstantOp>(label.getDefiningOp())) {
        if(auto strAttr = co.value().dyn_cast<StringAttr>()) {
            std::string label = strAttr.getValue().str();
            std::vector<std::string> * labels = ft.getLabels();
            if(labels) {
                // The column labels are known, so we search for the specified
                // label.
                std::vector<Type> colTypes = ft.getColumnTypes();
                for(size_t i = 0; i < colTypes.size(); i++)
                    if((*labels)[i] == label)
                        // Found the label.
                        return colTypes[i];
                // Did not find the label.
                throw std::runtime_error(
                        "the specified label was not found: '" + label + "'"
                );
            }
            else
                // The column labels are unknown, so we cannot tell what type
                // the column with the specified label has.
                return daphne::UnknownType::get(ft.getContext());
        }
    }
    throw std::runtime_error(
            "the specified label must be a constant of string type"
    );
}

// ****************************************************************************
// Type inference interface implementations
// ****************************************************************************

std::vector<Type> daphne::CastOp::inferTypes() {
    auto ftArg = arg().getType().dyn_cast<daphne::FrameType>();
    auto mtRes = res().getType().dyn_cast<daphne::MatrixType>();
    if(ftArg && mtRes && mtRes.getElementType().isa<daphne::UnknownType>()) {
        std::vector<Type> ctsArg = ftArg.getColumnTypes();
        if(ctsArg.size() == 1)
            return {daphne::MatrixType::get(getContext(), ctsArg[0])};
        else
            throw std::runtime_error(
                    "currently CastOp cannot infer the value type of its "
                    "output matrix, if the input is a multi-column frame"
            );
    }
    return {daphne::UnknownType::get(getContext())};
}

std::vector<Type> daphne::ExtractColOp::inferTypes() {
    auto ft = source().getType().dyn_cast<daphne::FrameType>();
    auto st = selectedCols().getType().dyn_cast<daphne::StringType>();
    if(ft && st) {
        Type vt = getFrameColumnTypeByLabel(ft, selectedCols());
        return {daphne::FrameType::get(getContext(), {vt})};
    }
    else
        throw std::runtime_error(
                "currently, ExtractColOp can only infer its type for frame "
                "inputs and a single column name"
        );
}

std::vector<Type> daphne::CreateFrameOp::inferTypes() {
    std::vector<Type> colTypes;
    for(Value col : cols())
        colTypes.push_back(col.getType().dyn_cast<daphne::MatrixType>().getElementType());
    return {daphne::FrameType::get(getContext(), colTypes)};
}

std::vector<Type> daphne::RandMatrixOp::inferTypes() {
    auto elTy = min().getType();
    if(elTy == UnknownType::get(getContext())) {
        elTy = max().getType();
    }
    else {
        assert((max().getType() == UnknownType::get(getContext()) || elTy == max().getType())
            && "Min and max need to have the same type");
    }
    return {daphne::MatrixType::get(getContext(), elTy)};
}

std::vector<Type> daphne::GroupJoinOp::inferTypes() {
    daphne::FrameType lhsFt = lhs().getType().dyn_cast<daphne::FrameType>();
    daphne::FrameType rhsFt = rhs().getType().dyn_cast<daphne::FrameType>();
    Type lhsOnType = getFrameColumnTypeByLabel(lhsFt, lhsOn());
    Type rhsAggType = getFrameColumnTypeByLabel(rhsFt, rhsAgg());

    MLIRContext * ctx = getContext();
    Builder builder(ctx);
    return {
        daphne::FrameType::get(ctx, {lhsOnType, rhsAggType}),
        daphne::MatrixType::get(ctx, builder.getIndexType())
    };
}

std::vector<Type> daphne::SemiJoinOp::inferTypes() {
    daphne::FrameType lhsFt = lhs().getType().dyn_cast<daphne::FrameType>();
    Type lhsOnType = getFrameColumnTypeByLabel(lhsFt, lhsOn());

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

    daphne::FrameType arg = frame().getType().dyn_cast<daphne::FrameType>();

    std::vector<Type> newColumnTypes;
    std::vector<Value> aggColValues;
    std::vector<std::string> aggFuncNames;

    for(Value t : keyCol()){
        //Key Types getting adopted for the new Frame
        newColumnTypes.push_back(getFrameColumnTypeByLabel(arg, t));
    }

    // Values get collected in a easier to use Datastructure
    for(Value t : aggCol()){
        aggColValues.push_back(t);
    }
    // Function names get collected in a easier to use Datastructure
    for(Attribute t: aggFuncs()){
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
    // TODO Take into accout if indexes or data shall be returned.
    Type srcType = arg().getType();
    Type t;
    if(auto mt = srcType.dyn_cast<daphne::MatrixType>())
        t = mt.withSameElementType();
    else if(auto ft = srcType.dyn_cast<daphne::FrameType>())
        t = ft.withSameColumnTypes();
    return {t};
}

std::vector<Type> daphne::SliceColOp::inferTypes() {
    throw std::runtime_error("type inference not implemented for SliceColOp"); // TODO
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
    std::vector<Type> types = daphne::tryInferType(op);
    const size_t numRes = op->getNumResults();
    if(types.size() != numRes)
        throw std::runtime_error(
                "type inference for op " +
                op->getName().getStringRef().str() + " returned " +
                std::to_string(types.size()) + " types, but the op has " +
                std::to_string(numRes) + " results"
        );
    // Set the infered types on all results of this operation.
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