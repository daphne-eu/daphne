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

#include <compiler/utils/CompilerUtils.h>

#include <mlir/IR/Value.h>

#include <string>

// **************************************************************************************************
// Specializations of isConstantHelper for string types
// **************************************************************************************************

template<>
std::pair<bool, std::string> CompilerUtils::isConstantHelper<std::string, mlir::StringAttr>(mlir::Value v, std::function<std::string(const mlir::StringAttr&)> func) {
    if(auto co = v.getDefiningOp<mlir::daphne::ConstantOp>()) {
        if(auto attr = co.getValue().dyn_cast<mlir::StringAttr>()) {
            return std::make_pair(true, func(attr));
        }
    }
    if(auto co = v.getDefiningOp<mlir::arith::ConstantOp>()) {
        if(auto attr = co.getValue().dyn_cast<mlir::StringAttr>()) {
            return std::make_pair(true, func(attr));
        }
    }
    return std::make_pair(false, std::string());
}

// **************************************************************************************************
// Specializations of isConstant for various types
// **************************************************************************************************

template<>
std::pair<bool, std::string> CompilerUtils::isConstant<std::string>(mlir::Value v) {
    return isConstantHelper<std::string, mlir::StringAttr>(
            v, [](mlir::StringAttr attr){return attr.getValue().str();}
    );
}

template<>
std::pair<bool, int64_t> CompilerUtils::isConstant<int64_t>(mlir::Value v) {
    return isConstantHelper<int64_t, mlir::IntegerAttr>(
            v, [](mlir::IntegerAttr attr){return attr.getValue().getLimitedValue();}
    );
}

template<>
std::pair<bool, float> CompilerUtils::isConstant<float>(mlir::Value v) {
    return isConstantHelper<float, mlir::FloatAttr>(
            v, [](mlir::FloatAttr attr){return attr.getValue().convertToFloat();}
    );
}

template<>
std::pair<bool, double> CompilerUtils::isConstant<double>(mlir::Value v) {
    return isConstantHelper<double, mlir::FloatAttr>(
            v, [](mlir::FloatAttr attr){return attr.getValue().convertToDouble();}
    );
}

template<>
std::pair<bool, bool> CompilerUtils::isConstant<bool>(mlir::Value v) {
    return isConstantHelper<bool, mlir::BoolAttr>(
            v, [](mlir::BoolAttr attr){return attr.getValue();}
    );
}

// **************************************************************************************************
// Specializations of constantOrThrow for various types
// **************************************************************************************************

template<>
std::string CompilerUtils::constantOrThrow<std::string>(mlir::Value v, const std::string & errorMsg) {
    return constantOrThrowHelper<std::string, mlir::StringAttr>(
            v, [](mlir::StringAttr attr){return attr.getValue().str();}, errorMsg, "string"
    );
}

template<>
int64_t CompilerUtils::constantOrThrow<int64_t>(mlir::Value v, const std::string & errorMsg) {
    return constantOrThrowHelper<int64_t, mlir::IntegerAttr>(
            v, [](mlir::IntegerAttr attr){return attr.getValue().getLimitedValue();}, errorMsg, "integer"
    );
}

template<>
uint64_t CompilerUtils::constantOrThrow<uint64_t>(mlir::Value v, const std::string & errorMsg) {
    return constantOrThrowHelper<uint64_t, mlir::IntegerAttr>(
            v, [](mlir::IntegerAttr attr){return attr.getValue().getLimitedValue();}, errorMsg, "integer"
    );
}

template<>
float CompilerUtils::constantOrThrow<float>(mlir::Value v, const std::string & errorMsg) {
    return constantOrThrowHelper<float, mlir::FloatAttr>(
            v, [](mlir::FloatAttr attr){return attr.getValue().convertToFloat();}, errorMsg, "float"
    );
}

template<>
double CompilerUtils::constantOrThrow<double>(mlir::Value v, const std::string & errorMsg) {
    return constantOrThrowHelper<double, mlir::FloatAttr>(
            v, [](mlir::FloatAttr attr){return attr.getValue().convertToDouble();}, errorMsg, "double"
    );
}

template<>
bool CompilerUtils::constantOrThrow<bool>(mlir::Value v, const std::string & errorMsg) {
    return constantOrThrowHelper<bool, mlir::BoolAttr>(
            v, [](mlir::BoolAttr attr){return attr.getValue();}, errorMsg, "bool"
    );
}

// **************************************************************************************************
// Specializations of constantOrDefault for various types
// **************************************************************************************************

template<>
std::string CompilerUtils::constantOrDefault<std::string>(mlir::Value v, std::string d) {
    return constantOrDefaultHelper<std::string, mlir::StringAttr>(
            v, d, [](mlir::StringAttr attr){return attr.getValue().str();}
    );
}

template<>
int64_t CompilerUtils::constantOrDefault<int64_t>(mlir::Value v, int64_t d) {
    return constantOrDefaultHelper<int64_t, mlir::IntegerAttr>(
            v, d, [](mlir::IntegerAttr attr){return attr.getValue().getLimitedValue();}
    );
}

template<>
float CompilerUtils::constantOrDefault<float>(mlir::Value v, float d) {
    return constantOrDefaultHelper<float, mlir::FloatAttr>(
            v, d, [](mlir::FloatAttr attr){return attr.getValue().convertToFloat();}
    );
}

template<>
double CompilerUtils::constantOrDefault<double>(mlir::Value v, double d) {
    return constantOrDefaultHelper<double, mlir::FloatAttr>(
            v, d, [](mlir::FloatAttr attr){return attr.getValue().convertToDouble();}
    );
}

template<>
bool CompilerUtils::constantOrDefault<bool>(mlir::Value v, bool d) {
    return constantOrDefaultHelper<bool, mlir::BoolAttr>(
            v, d, [](mlir::BoolAttr attr){return attr.getValue();}
    );
}

// **************************************************************************************************
// Other
// **************************************************************************************************

[[maybe_unused]] FileMetaData CompilerUtils::getFileMetaData(mlir::Value filename) {
    return MetaDataParser::readMetaData(constantOrThrow<std::string>(filename));
}

bool CompilerUtils::isMatrixComputation(mlir::Operation *v) {
    return
            llvm::any_of(v->getOperandTypes(), [&](mlir::Type ty){ return ty.isa<mlir::daphne::MatrixType>(); })
            ||
            llvm::any_of(v->getResultTypes(), [&](mlir::Type ty){ return ty.isa<mlir::daphne::MatrixType>(); });
}
