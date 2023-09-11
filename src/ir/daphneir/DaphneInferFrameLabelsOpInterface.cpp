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
#include <ir/daphneir/Daphne.h>
#include <runtime/local/datastructures/LabelUtils.h>

#include <mlir/IR/Value.h>

#include <string>
#include <vector>
#include <stdexcept>

namespace mlir::daphne {
#include <ir/daphneir/DaphneInferFrameLabelsOpInterface.cpp.inc>
}

using namespace mlir;

// ****************************************************************************
// Frame label inference utility functions
// ****************************************************************************
// For families of operations.

template<class ExtractOrFilterRowOp>
void inferFrameLabels_ExtractOrFilterRowOp(ExtractOrFilterRowOp * op) {
    Type t = op->getSource().getType();
    if(auto ft = t.dyn_cast<daphne::FrameType>()) {
        Value res = op->getResult();
        res.setType(res.getType().dyn_cast<daphne::FrameType>().withLabels(ft.getLabels()));
    }
}

// ****************************************************************************
// Frame label inference implementations
// ****************************************************************************

void daphne::ReadOp::inferFrameLabels() {
    auto p = CompilerUtils::isConstant<std::string>(getFileName());
    if (auto resType = getRes().getType().dyn_cast<daphne::FrameType>()) {
        if (p.first) {
            std::vector<std::string> * labels;
            FileMetaData fmd = CompilerUtils::getFileMetaData(getFileName());
            if (fmd.labels.empty()) {
                labels = nullptr;
            } else {
                labels = new std::vector<std::string>(fmd.labels);
            }

            Value res = getResult();
            res.setType(res.getType().dyn_cast<daphne::FrameType>().withLabels(labels));
        }
    }
}

void daphne::ColBindOp::inferFrameLabels() {
    auto ftLhs = getLhs().getType().dyn_cast<daphne::FrameType>();
    auto ftRhs = getRhs().getType().dyn_cast<daphne::FrameType>();

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
    res.setType(res.getType().dyn_cast<daphne::FrameType>().withLabels(labelsRes));
}

void daphne::CreateFrameOp::inferFrameLabels() {
    auto resLabels = new std::vector<std::string>();
    for(Value label : getLabels())
        resLabels->push_back(CompilerUtils::constantOrThrow<std::string>(label));
    Value res = getResult();
    res.setType(res.getType().dyn_cast<daphne::FrameType>().withLabels(resLabels));
}

void daphne::ExtractColOp::inferFrameLabels() {
    auto ft = getSource().getType().dyn_cast<daphne::FrameType>();
    auto st = getSelectedCols().getType().dyn_cast<daphne::StringType>();
    if(ft && st) {
        auto resLabels = new std::vector<std::string>();
        resLabels->push_back(CompilerUtils::constantOrThrow<std::string>(getSelectedCols()));
        Value res = getResult();
        res.setType(res.getType().dyn_cast<daphne::FrameType>().withLabels(resLabels));
    }
}

void daphne::ExtractRowOp::inferFrameLabels() {
    inferFrameLabels_ExtractOrFilterRowOp(this);
}

void daphne::FilterRowOp::inferFrameLabels() {
    inferFrameLabels_ExtractOrFilterRowOp(this);
}

void daphne::GroupJoinOp::inferFrameLabels() {
    auto newLabels = new std::vector<std::string>();
    newLabels->push_back(CompilerUtils::constantOrThrow<std::string>(getLhsOn()));
    newLabels->push_back(std::string("SUM(") + CompilerUtils::constantOrThrow<std::string>(getRhsAgg()) + std::string(")"));
    Value res = getResult(0);
    res.setType(res.getType().dyn_cast<daphne::FrameType>().withLabels(newLabels));
}

void daphne::SemiJoinOp::inferFrameLabels() {
    auto newLabels = new std::vector<std::string>();
    newLabels->push_back(CompilerUtils::constantOrThrow<std::string>(getLhsOn()));
    Value res = getResult(0);
    res.setType(res.getType().dyn_cast<daphne::FrameType>().withLabels(newLabels));
}

void daphne::CartesianOp::inferFrameLabels() {
    auto newLabels = new std::vector<std::string>();
    auto ft1 = getLhs().getType().dyn_cast<daphne::FrameType>();
    auto ft2 = getRhs().getType().dyn_cast<daphne::FrameType>();
    std::vector<std::string> * labelsStr1 = ft1.getLabels();
    std::vector<std::string> * labelsStr2 = ft2.getLabels();

    if(labelsStr1)
        for(auto labelStr : *labelsStr1)
            newLabels->push_back(labelStr);
    if(labelsStr2)
        for(auto labelStr : *labelsStr2)
            newLabels->push_back(labelStr);

    getResult().setType(getRes().getType().dyn_cast<daphne::FrameType>().withLabels(newLabels));
}

void daphne::OrderOp::inferFrameLabels() {
    Type t = getArg().getType();
    if(auto ft = t.dyn_cast<daphne::FrameType>()) {
        Value res = getResult();
        res.setType(res.getType().dyn_cast<daphne::FrameType>().withLabels(ft.getLabels()));
    }
}

void daphne::InnerJoinOp::inferFrameLabels() {
    auto newLabels = new std::vector<std::string>();
    auto ft1 = getLhs().getType().dyn_cast<daphne::FrameType>();
    auto ft2 = getRhs().getType().dyn_cast<daphne::FrameType>();
    std::vector<std::string> * labelsStr1 = ft1.getLabels();
    std::vector<std::string> * labelsStr2 = ft2.getLabels();

    if(labelsStr1)
        for(auto labelStr : *labelsStr1)
            newLabels->push_back(labelStr);
    if(labelsStr2)
        for(auto labelStr : *labelsStr2)
            newLabels->push_back(labelStr);

    getResult().setType(getRes().getType().dyn_cast<daphne::FrameType>().withLabels(newLabels));
}

void daphne::ThetaJoinOp::inferFrameLabels() {
    std::vector<std::string> * newLabels = nullptr;
    
    auto ft1 = getLhs().getType().dyn_cast<daphne::FrameType>();
    auto ft2 = getRhs().getType().dyn_cast<daphne::FrameType>();
    std::vector<std::string> * labelsStr1 = ft1.getLabels();
    std::vector<std::string> * labelsStr2 = ft2.getLabels();

    if(labelsStr1 && labelsStr2) {
        newLabels = new std::vector<std::string>();
        for(auto labelStr : *labelsStr1)
            newLabels->push_back(labelStr);
        for(auto labelStr : *labelsStr2)
            newLabels->push_back(labelStr);
    }

    getResult().setType(getRes().getType().dyn_cast<daphne::FrameType>().withLabels(newLabels));
}

void daphne::GroupOp::inferFrameLabels() {
    auto newLabels = new std::vector<std::string>();
    std::vector<std::string> aggColLabels;
    std::vector<std::string> aggFuncNames;

    for(Value t: getKeyCol()){ //Adopting keyCol Labels
        newLabels->push_back(CompilerUtils::constantOrThrow<std::string>(t));
    }

    for(Value t: getAggCol()){
        aggColLabels.push_back(CompilerUtils::constantOrThrow<std::string>(t));
    }
    for(Attribute t: getAggFuncs()){
        GroupEnum aggFuncValue = t.dyn_cast<GroupEnumAttr>().getValue();
        aggFuncNames.push_back(stringifyGroupEnum(aggFuncValue).str());
    }
    for(size_t i = 0; i < aggFuncNames.size() && i < aggColLabels.size(); i++){
        newLabels->push_back(aggFuncNames.at(i) + "(" + aggColLabels.at(i) + ")");
    }

    getResult().setType(getRes().getType().dyn_cast<daphne::FrameType>().withLabels(newLabels));
}

void daphne::SetColLabelsOp::inferFrameLabels() {
    auto newLabels = new std::vector<std::string>();
    for(Value label : getLabels()) {
        try {
            newLabels->push_back(CompilerUtils::constantOrThrow<std::string>(label));
        }
        catch(std::runtime_error&) {
            // TODO This could be improved by supporting knowledge on only some
            // of the labels.
            // If we do not know the values of all label operands at
            // compile-time, then we do not infer any of them.
            delete newLabels;
            newLabels = nullptr;
        }
    }
    getResult().setType(getRes().getType().dyn_cast<daphne::FrameType>().withLabels(newLabels));
}

void daphne::SetColLabelsPrefixOp::inferFrameLabels() {
    auto newLabels = new std::vector<std::string>();
    std::string prefixStr = CompilerUtils::constantOrThrow<std::string>(getPrefix());
    auto ft = getArg().getType().dyn_cast<daphne::FrameType>();
    std::vector<std::string> * labelsStr = ft.getLabels();
    if(labelsStr)
        for(auto labelStr : *labelsStr)
            newLabels->push_back(LabelUtils::setPrefix(prefixStr, labelStr));
    else {
        delete newLabels;
        newLabels = nullptr;
    }
    getResult().setType(getRes().getType().dyn_cast<daphne::FrameType>().withLabels(newLabels));
}
