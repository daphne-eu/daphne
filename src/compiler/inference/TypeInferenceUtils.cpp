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

#include <compiler/inference/TypeInferenceUtils.h>

int generality(mlir::Type t) {
    using namespace mlir;
    
    // TODO It is debatable if unsigned int shall be more general than signed
    // int of the same bit width.
    
    // The greater the number, the more general the type.
    if(t.isa<daphne::UnknownType>()) return 11;
    if(t.isa<daphne::StringType>()) return 10;
    if(t.isF64()) return 9;
    if(t.isF32()) return 8;
    if(t.isUnsignedInteger(64)) return 7;
    if(t.  isSignedInteger(64)) return 6;
    if(t.isIndex()) return 5;
    if(t.isUnsignedInteger(32)) return 4;
    if(t.  isSignedInteger(32)) return 3;
    if(t.isUnsignedInteger(8)) return 2;
    if(t.  isSignedInteger(8)) return 1;
    if(t.        isInteger(1)) return 0;
    
    std::string str;
    llvm::raw_string_ostream msg(str);
    msg << "no generality code available for value type: " << t;
    throw std::runtime_error(msg.str());
}

mlir::Type mostGeneralVt(const std::vector<mlir::Type> & vt) {
    if(vt.empty())
        throw std::runtime_error(
                "mostGeneralVt() invoked with empty list of value types"
        );
                
    mlir::Type res = vt[0];
    for(size_t i = 1; i < vt.size(); i++)
        if(generality(vt[i]) > generality(res))
            res = vt[i];
    
    return res;
}

mlir::Type mostGeneralVt(const std::vector<std::vector<mlir::Type>> & vts, size_t num) {
    if(vts.empty())
        throw std::runtime_error(
                "mostGeneralVt() invoked with empty list of lists of value types"
        );
    
    if(num == 0)
        num = vts.size();
    
    mlir::Type res = mostGeneralVt(vts[0]);
    for(size_t i = 1; i < std::min(vts.size(), num); i++) {
        mlir::Type cur = mostGeneralVt(vts[i]);
        if(generality(cur) > generality(res))
            res = cur;
    }
    
    return res;
}

std::vector<mlir::Type> inferValueTypeFromArgs(
        const std::vector<DataTypeCode> & argDtc,
        std::vector<std::vector<mlir::Type>> & argVts
) {
    // TODO Simplify: resDtc is already known. If it's not Frame, this
    // can be done simpler and we don't need the getMostGeneralVt later.

    // Find out if we need to expand the value type of matrix and scalar
    // arguments to match the number of column types of frame arguments.
    size_t commonNumFrameCols = 1;
    bool hasFrame = false;
    for(size_t i = 0; i < argVts.size(); i++)
        if(argDtc[i] == DataTypeCode::FRAME) {
            if(hasFrame && argVts[i].size() != commonNumFrameCols)
                throw std::runtime_error(
                        "type inference trait ValueTypeFromArgs requires that "
                        "all input frames have the same number of columns"
                );
            hasFrame = true;
            commonNumFrameCols = argVts[i].size();
        }

    // If required: Expand the value type of matrix and scalar arguments to
    // match the common number of column types of frame arguments.
    if(hasFrame)
        for(size_t i = 0; i < argVts.size(); i++)
            if(argDtc[i] != DataTypeCode::FRAME)
                argVts[i] = std::vector(commonNumFrameCols, argVts[i][0]);

    // Determine the most general argument value type. This is done for each
    // column separately, if frames are involved.
    std::vector<mlir::Type> resVts = argVts[0];
    for(size_t i = 1; i < argVts.size(); i++)
        for(size_t k = 0; k < commonNumFrameCols; k++)
            if(generality(argVts[i][k]) > generality(resVts[k]))
                resVts[k] = argVts[i][k];
    
    return resVts;
}