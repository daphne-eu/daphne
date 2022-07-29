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

using namespace mlir;

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
// Type inference utility functions
// ****************************************************************************
// For families of operations.

template<class EwCmpOp>
void inferTypes_EwCmpOp(EwCmpOp * op) {
    Type lhsType = op->lhs().getType();
    Type rhsType = op->rhs().getType();
    Type t;
    if(auto mt = lhsType.dyn_cast<daphne::MatrixType>())
        t = mt.withSameElementType();
    else if(auto mt = rhsType.dyn_cast<daphne::MatrixType>())
        t = mt.withSameElementType();
    else {
        // TODO: check rhsType?
        // same as input type, not bool (design decision)
        t = lhsType;
    }
    op->getResult().setType(t);
}

template<class EwArithOp>
void inferTypes_EwArithOp(EwArithOp * op) {
    Type lhsType = op->lhs().getType();
    Type rhsType = op->rhs().getType();
    Type t;
    if(auto mt = lhsType.dyn_cast<daphne::MatrixType>())
        t = mt.withSameElementType();
    else if(auto mt = rhsType.dyn_cast<daphne::MatrixType>())
        t = mt.withSameElementType();
    else {
        // TODO: check rhsType?
        t = lhsType;
    }
    op->getResult().setType(t);
}

template<class AllAggOp>
void inferTypes_AllAggOp(AllAggOp * op) {
    Type argType = op->arg().getType();
    // TODO: f64, si64 and ui64 for sum?
    op->getResult().setType(argType.cast<daphne::MatrixType>().getElementType());
}

// ****************************************************************************
// Type inference implementations
// ****************************************************************************

void daphne::CastOp::inferTypes() {
    auto ftArg = arg().getType().dyn_cast<daphne::FrameType>();
    auto mtRes = res().getType().dyn_cast<daphne::MatrixType>();
    if(ftArg && mtRes && mtRes.getElementType().isa<daphne::UnknownType>()) {
        std::vector<Type> ctsArg = ftArg.getColumnTypes();
        if(ctsArg.size() == 1)
            res().setType(daphne::MatrixType::get(getContext(), ctsArg[0]));
        else
            throw std::runtime_error(
                    "currently CastOp cannot infer the value type of its "
                    "output matrix, if the input is a multi-column frame"
            );
    }
}

void daphne::ColBindOp::inferTypes() {
    auto ftLhs = lhs().getType().dyn_cast<daphne::FrameType>();
    auto ftRhs = rhs().getType().dyn_cast<daphne::FrameType>();
    if(ftLhs && ftRhs) {
        std::vector<Type> newColumnTypes;
        for(Type t : ftLhs.getColumnTypes())
            newColumnTypes.push_back(t);
        for(Type t : ftRhs.getColumnTypes())
            newColumnTypes.push_back(t);
        getResult().setType(
                daphne::FrameType::get(getContext(), newColumnTypes)
        );
    }
}

void daphne::ExtractColOp::inferTypes() {
    auto ft = source().getType().dyn_cast<daphne::FrameType>();
    auto st = selectedCols().getType().dyn_cast<daphne::StringType>();
    if(ft && st) {
        Type vt = getFrameColumnTypeByLabel(ft, selectedCols());
        getResult().setType(daphne::FrameType::get(getContext(), {vt}));
    }
    else
        throw std::runtime_error(
                "currently, ExtractColOp can only infer its type for frame "
                "inputs and a single column name"
        );
}

void daphne::CreateFrameOp::inferTypes() {
    std::vector<Type> colTypes;
    for(Value col : cols())
        colTypes.push_back(col.getType().dyn_cast<daphne::MatrixType>().getElementType());
    getResult().setType(daphne::FrameType::get(getContext(), colTypes));
}

void daphne::RandMatrixOp::inferTypes() {
    auto elTy = min().getType();
    if(elTy == UnknownType::get(getContext())) {
        elTy = max().getType();
    }
    else {
        assert((max().getType() == UnknownType::get(getContext()) || elTy == max().getType())
            && "Min and max need to have the same type");
    }
    getResult().setType(daphne::MatrixType::get(getContext(), elTy));
}

void daphne::EwEqOp::inferTypes() {
    return inferTypes_EwCmpOp(this);
}

void daphne::EwNeqOp::inferTypes() {
    return inferTypes_EwCmpOp(this);
}

void daphne::EwLtOp::inferTypes() {
    return inferTypes_EwCmpOp(this);
}

void daphne::EwLeOp::inferTypes() {
    return inferTypes_EwCmpOp(this);
}

void daphne::EwGtOp::inferTypes() {
    return inferTypes_EwCmpOp(this);
}

void daphne::EwGeOp::inferTypes() {
    return inferTypes_EwCmpOp(this);
}

void daphne::ExtractRowOp::inferTypes() {
    Type srcType = source().getType();
    Type t;
    if(auto mt = srcType.dyn_cast<daphne::MatrixType>())
        t = mt.withSameElementType();
    else if(auto ft = srcType.dyn_cast<daphne::FrameType>())
        t = ft.withSameColumnTypes();
    getResult().setType(t);
}

void daphne::MatMulOp::inferTypes() {
    getResult().setType(lhs().getType().dyn_cast<daphne::MatrixType>().withSameElementType());
}

void daphne::FilterRowOp::inferTypes() {
    Type srcType = source().getType();
    Type t;
    if(auto mt = srcType.dyn_cast<daphne::MatrixType>())
        t = mt.withSameElementType();
    else if(auto ft = srcType.dyn_cast<daphne::FrameType>())
        t = ft.withSameColumnTypes();
    getResult().setType(t);
}

void daphne::GroupJoinOp::inferTypes() {
    daphne::FrameType lhsFt = lhs().getType().dyn_cast<daphne::FrameType>();
    daphne::FrameType rhsFt = rhs().getType().dyn_cast<daphne::FrameType>();
    Type lhsOnType = getFrameColumnTypeByLabel(lhsFt, lhsOn());
    Type rhsAggType = getFrameColumnTypeByLabel(rhsFt, rhsAgg());

    MLIRContext * ctx = getContext();
    Builder builder(ctx);
    getResult(0).setType(daphne::FrameType::get(ctx, {lhsOnType, rhsAggType}));
    getResult(1).setType(daphne::MatrixType::get(ctx, builder.getIndexType()));
}

void daphne::SemiJoinOp::inferTypes() {
    daphne::FrameType lhsFt = lhs().getType().dyn_cast<daphne::FrameType>();
    Type lhsOnType = getFrameColumnTypeByLabel(lhsFt, lhsOn());

    MLIRContext * ctx = getContext();
    Builder builder(ctx);
    getResult(0).setType(daphne::FrameType::get(ctx, {lhsOnType}));
    getResult(1).setType(daphne::MatrixType::get(ctx, builder.getIndexType()));
}

void daphne::InnerJoinOp::inferTypes() {
    daphne::FrameType ftLhs = lhs().getType().dyn_cast<daphne::FrameType>();
    daphne::FrameType ftRhs = rhs().getType().dyn_cast<daphne::FrameType>();
    if(ftLhs && ftRhs) {
        std::vector<Type> newColumnTypes;
        for(Type t : ftLhs.getColumnTypes())
            newColumnTypes.push_back(t);
        for(Type t : ftRhs.getColumnTypes())
            newColumnTypes.push_back(t);
        getResult().setType(
                daphne::FrameType::get(getContext(), newColumnTypes)
        );
    }
}

void daphne::GroupOp::inferTypes() {
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
    getResult().setType(daphne::FrameType::get(ctx, newColumnTypes));
}

void daphne::SetColLabelsOp::inferTypes() {
    getResult().setType(
            arg().getType().dyn_cast<daphne::FrameType>().withSameColumnTypes()
    );
}

void daphne::SetColLabelsPrefixOp::inferTypes() {
    getResult().setType(
            arg().getType().dyn_cast<daphne::FrameType>().withSameColumnTypes()
    );
}

void daphne::AllAggMaxOp::inferTypes() {
    return inferTypes_AllAggOp(this);
}

void daphne::AllAggMeanOp::inferTypes() {
    return inferTypes_AllAggOp(this);
}

void daphne::AllAggMinOp::inferTypes() {
    return inferTypes_AllAggOp(this);
}

void daphne::AllAggStddevOp::inferTypes() {
    return inferTypes_AllAggOp(this);
}

void daphne::AllAggSumOp::inferTypes() {
    return inferTypes_AllAggOp(this);
}

void daphne::AllAggVarOp::inferTypes() {
    return inferTypes_AllAggOp(this);
}

void daphne::EwAddOp::inferTypes() {
    return inferTypes_EwArithOp(this);
}

void daphne::EwAndOp::inferTypes() {
    return inferTypes_EwArithOp(this);
}

void daphne::EwConcatOp::inferTypes() {
    getResult().setType(StringType::get(getContext()));
}

void daphne::EwDivOp::inferTypes() {
    return inferTypes_EwArithOp(this);
}

void daphne::EwLogOp::inferTypes() {
    return inferTypes_EwArithOp(this);
}

void daphne::EwMaxOp::inferTypes() {
    return inferTypes_EwArithOp(this);
}

void daphne::EwMinOp::inferTypes() {
    return inferTypes_EwArithOp(this);
}

void daphne::EwModOp::inferTypes() {
    return inferTypes_EwArithOp(this);
}

void daphne::EwMulOp::inferTypes() {
    return inferTypes_EwArithOp(this);
}

void daphne::EwOrOp::inferTypes() {
    return inferTypes_EwArithOp(this);
}

void daphne::EwPowOp::inferTypes() {
    return inferTypes_EwArithOp(this);
}

void daphne::EwSubOp::inferTypes() {
    return inferTypes_EwArithOp(this);
}

void daphne::EwXorOp::inferTypes() {
    return inferTypes_EwArithOp(this);
}
