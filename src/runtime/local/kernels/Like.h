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

#pragma once

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <string_view>
#include <vector>
#include <regex>

using StringMatrix = DenseMatrix<const char*>;
const char wildCardAnyNumberChars = '%';
const char wildCardSingleChar = '_';
const char matchWildCard[] = {wildCardAnyNumberChars, wildCardSingleChar, '\0'};

struct Like {
    static std::regex createRegex(std::string_view pattern){
        std::string expression(pattern);
        for(size_t charIdx = 0; charIdx < expression.size(); charIdx++){
            if(expression[charIdx] == wildCardAnyNumberChars){
                expression.replace(charIdx, 1, ".*", 2);
                charIdx++;
            }else if(expression[charIdx] == wildCardSingleChar)
                expression[charIdx] = '.';
        }
        return std::regex(expression);
    }

    static void apply(StringMatrix *& res, const StringMatrix * arg, const size_t colIdx, std::string_view pattern, DCTX(ctx)) {
        const size_t argRows = arg->getNumRows();
        const size_t argCols = arg->getNumCols();
        if(colIdx > argCols)
            throw std::runtime_error("like: invalid column index");

        auto valuesArg = arg->getValues();
        std::vector<size_t> argRowMatchIdxs;
        const std::regex exprToMatch = createRegex(pattern);

        for(size_t r = 0; r < argRows; r++) {
            if(std::regex_match(valuesArg[colIdx], exprToMatch))
                argRowMatchIdxs.push_back(r);
            valuesArg += arg->getRowSkip();
        }
        
        if(res == nullptr)
            res = DataObjectFactory::create<StringMatrix>(argRowMatchIdxs.size(), argCols, false);
    
        for(size_t resRowIdx = 0; resRowIdx < argRowMatchIdxs.size(); resRowIdx++)
            for(size_t c = 0; c < argCols; c++)
                res->set(resRowIdx, c, arg->get(argRowMatchIdxs[resRowIdx], c));
    }
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

void like(StringMatrix *& res, const StringMatrix * arg, const size_t colIdx, std::string_view pattern, DCTX(ctx)) {
    Like::apply(res, arg, colIdx, pattern, ctx);
}
