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

#include <mlir/IR/Value.h>

#include <string>
#include <vector>
#include <stdexcept>

namespace mlir::daphne {
#include <ir/daphneir/DaphneInferFrameLabelsOpInterface.cpp.inc>
}

using namespace mlir;

// ****************************************************************************
// General utility fuctions
// ****************************************************************************

// TODO This could become a general utility (generalize for different types, 
// also useful in the parser and other compiler passes).
std::string getConstantString(Value v) {
    if(auto co = llvm::dyn_cast<daphne::ConstantOp>(v.getDefiningOp()))
        if(auto strAttr = co.value().dyn_cast<StringAttr>())
            return strAttr.getValue().str();
    throw std::runtime_error(
            "the given value must be a constant of string type"
    );
}

// ****************************************************************************
// Frame label inference utility functions
// ****************************************************************************
// For families of operations.

template<class ExtractOrFilterRowOp>
void inferFrameLabels_ExtractOrFilterRowOp(ExtractOrFilterRowOp * op) {
    Type t = op->source().getType();
    if(auto ft = t.dyn_cast<daphne::FrameType>()) {
        Value res = op->getResult();
        auto oldResType = res.getType().dyn_cast<daphne::FrameType>();
        auto newResType = daphne::FrameType::get(
                op->getContext(),
                oldResType.getColumnTypes(),
                ft.getLabels()
        );
        res.setType(newResType);
    }
}

// ****************************************************************************
// Frame label inference implementations
// ****************************************************************************

void daphne::ColBindOp::inferFrameLabels() {
    auto ftLhs = lhs().getType().dyn_cast<daphne::FrameType>();
    auto ftRhs = rhs().getType().dyn_cast<daphne::FrameType>();
    
    if(!ftLhs || !ftRhs)
        throw std::runtime_error(
                "currently ColBindOp can only infer its output labels if both "
                "inputs are frames"
        );
    if(!ftLhs.getLabels() || !ftRhs.getLabels())
        throw std::runtime_error(
                "currenly ColBindOp can only infer its output labels if the "
                "labels of both input frames are known"
        );
    
    auto labelsRes = new std::vector<std::string>();
    for(auto l : *(ftLhs.getLabels()))
        labelsRes->push_back(l);
    for(auto l : *(ftRhs.getLabels()))
        labelsRes->push_back(l);
    
    Value res = getResult();
    auto oldResType = res.getType().dyn_cast<daphne::FrameType>();
    auto newResType = daphne::FrameType::get(
            getContext(),
            oldResType.getColumnTypes(),
            labelsRes
    );
    res.setType(newResType);
}

void daphne::CreateFrameOp::inferFrameLabels() {
    auto resLabels = new std::vector<std::string>();
    for(Value label : labels())
        resLabels->push_back(getConstantString(label));
    Value res = getResult();
    auto oldResType = res.getType().dyn_cast<daphne::FrameType>();
    auto newResType = daphne::FrameType::get(
            getContext(),
            oldResType.getColumnTypes(),
            resLabels
    );
    res.setType(newResType);
}

void daphne::ExtractColOp::inferFrameLabels() {
    auto ft = source().getType().dyn_cast<daphne::FrameType>();
    auto st = selectedCols().getType().dyn_cast<daphne::StringType>();
    if(ft && st) {
        auto resLabels = new std::vector<std::string>();
        resLabels->push_back(getConstantString(selectedCols()));
        
        Value res = getResult();
        auto oldResType = res.getType().dyn_cast<daphne::FrameType>();
        auto newResType = daphne::FrameType::get(
                getContext(),
                oldResType.getColumnTypes(),
                resLabels
        );
        res.setType(newResType);
    }
    else
        throw std::runtime_error(
                "currently ExtractColOp can only infer its output labels for "
                "frame inputs and a single column name"
        );
}

void daphne::ExtractRowOp::inferFrameLabels() {
    inferFrameLabels_ExtractOrFilterRowOp(this);
}

void daphne::FilterRowOp::inferFrameLabels() {
    inferFrameLabels_ExtractOrFilterRowOp(this);
}

void daphne::GroupJoinOp::inferFrameLabels() {
    Value res = getResult(0);
    auto ft = res.getType().dyn_cast<daphne::FrameType>();
    auto newLabels = new std::vector<std::string>();
    newLabels->push_back(getConstantString(lhsOn()));
    newLabels->push_back(getConstantString(rhsAgg()));
    res.setType(daphne::FrameType::get(
            getContext(), ft.getColumnTypes(), newLabels
    ));
}

void daphne::SemiJoinOp::inferFrameLabels() {
    Value res = getResult(0);
    auto ft = res.getType().dyn_cast<daphne::FrameType>();
    auto newLabels = new std::vector<std::string>();
    newLabels->push_back(getConstantString(lhsOn()));
    res.setType(daphne::FrameType::get(
            getContext(), ft.getColumnTypes(), newLabels
    ));
}