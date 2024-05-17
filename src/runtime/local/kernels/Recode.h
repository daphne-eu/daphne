/*
 * Copyright 2024 The DAPHNE Consortium
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
#include <runtime/local/datastructures/Matrix.h>

#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <set>
#include <vector>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTDict, class DTArg>
struct Recode {
    static void apply(DTRes *& res, DTDict *& dict, const DTArg * arg, bool orderPreserving, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTDict, class DTArg>
void recode(DTRes *& res, DTDict *& dict, const DTArg * arg, bool orderPreserving, DCTX(ctx)) {
    Recode<DTRes, DTDict, DTArg>::apply(res, dict, arg, orderPreserving, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template<typename VTVal, typename VTCode>
struct Recode<DenseMatrix<VTCode>, DenseMatrix<VTVal>, DenseMatrix<VTVal>> {
    static void apply(DenseMatrix<VTCode> *& res, DenseMatrix<VTVal> *& dict, const DenseMatrix<VTVal> * arg, bool orderPreserving, DCTX(ctx)) {
        // Validation.
        // TODO Remove this requirement, it's not strictly necessary.
        if(arg->getNumCols() != 1)
            throw std::runtime_error("recode: the argument must have exactly one column");
        
        if(orderPreserving) {
            // Determine the distinct values in the input.
            std::set<VTVal> distinct;
            const size_t numRowsArg = arg->getNumRows();
            const VTVal * valuesArg = arg->getValues();
            const size_t rowSkipArg = arg->getRowSkip();
            for(size_t r = 0; r < numRowsArg; r++) {
                distinct.insert(*valuesArg);
                valuesArg += rowSkipArg;
            }

            // Allocate output for the decoding dictionary.
            if(dict == nullptr)
                dict = DataObjectFactory::create<DenseMatrix<VTVal>>(distinct.size(), 1, false);

            // Create recoding dictionary and store decoding dictionary.
            std::unordered_map<VTVal, VTCode> recodeDict;
            VTVal * valuesDict = dict->getValues();
            const size_t rowSkipDict = dict->getRowSkip();
            VTCode nextCode = 0;
            for(auto it = distinct.begin(); it != distinct.end(); it++) {
                recodeDict.emplace(*it, nextCode++);
                *valuesDict = *it;
                valuesDict += rowSkipDict;
            }

            // Allocate output for recoded data.
            if(res == nullptr)
                res = DataObjectFactory::create<DenseMatrix<VTCode>>(numRowsArg, 1, false);

            // Recode the data.
            valuesArg = arg->getValues(); // rewind
            VTCode * valuesRes = res->getValues();
            const size_t rowSkipRes = res->getRowSkip();
            for(size_t r = 0; r < numRowsArg; r++) {
                *valuesRes = recodeDict[*valuesArg];
                valuesArg += rowSkipArg;
                valuesRes += rowSkipRes;
            }
        }
        else {
            const size_t numRowsArg = arg->getNumRows();

            // Allocate output for recoded data.
            if(res == nullptr)
                res = DataObjectFactory::create<DenseMatrix<VTCode>>(numRowsArg, 1, false);
            
            // Internal data structure.
            VTCode nextCode = 0;
            std::unordered_map<VTVal, VTCode> recodeDict;

            // Recode the data.
            const VTVal * valuesArg = arg->getValues();
            const size_t rowSkipArg = arg->getRowSkip();
            VTCode * valuesRes = res->getValues();
            const size_t rowSkipRes = res->getRowSkip();
            for(size_t r = 0; r < numRowsArg; r++) {
                const VTVal v = *valuesArg;

                auto it = recodeDict.find(v);
                if(it != recodeDict.end())
                    *valuesRes = it->second;
                else {
                    recodeDict[v] = nextCode;
                    *valuesRes = nextCode;
                    nextCode++;
                }

                valuesArg += rowSkipArg;
                valuesRes += rowSkipRes;
            }

            // Allocate output for the decoding dictionary.
            if(dict == nullptr)
                dict = DataObjectFactory::create<DenseMatrix<VTVal>>(recodeDict.size(), 1, false);

            // Store decoding dictionary.
            VTVal * valuesDict = dict->getValues();
            const size_t rowSkipDict = dict->getRowSkip();
            for(auto it = recodeDict.begin(); it != recodeDict.end(); it++) {
                const VTVal v = it->first;
                const VTCode c = it->second;
                valuesDict[c * rowSkipDict] = v;
            }
        }
    }
};

// ----------------------------------------------------------------------------
// Matrix <- Matrix
// ----------------------------------------------------------------------------

template<typename VTVal, typename VTCode>
struct Recode<Matrix<VTCode>, Matrix<VTVal>, Matrix<VTVal>> {
    static void apply(Matrix<VTCode> *& res, Matrix<VTVal> *& dict, const Matrix<VTVal> * arg, bool orderPreserving, DCTX(ctx)) {
        // Validation.
        // TODO Remove this requirement, it's not strictly necessary.
        if (arg->getNumCols() != 1)
            throw std::runtime_error("recode: the argument must have exactly one column");

        const size_t numRowsArg = arg->getNumRows();
        
        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTCode>>(numRowsArg, 1, false);
        

        if (orderPreserving) {
            // Determine the distinct values in the input.
            std::set<VTVal> distinct;
            for (size_t r = 0; r < numRowsArg; ++r) {
                distinct.insert(arg->get(r, 0));
            }

            // Allocate output for the decoding dictionary.
            if (dict == nullptr)
                dict = DataObjectFactory::create<DenseMatrix<VTVal>>(distinct.size(), 1, false);

            // Create recoding dictionary and store decoding dictionary.
            VTCode nextCode = 0;
            std::unordered_map<VTVal, VTCode> recodeDict;

            dict->prepareAppend();
            for (auto it = distinct.begin(); it != distinct.end(); ++it) {
                recodeDict.emplace(*it, nextCode);
                dict->append(nextCode++, 0, *it);
            }
            dict->finishAppend();

            // Recode the data.
            res->prepareAppend();
            for (size_t r = 0; r < numRowsArg; ++r)
                res->append(r, 0, recodeDict[arg->get(r, 0)]);
            res->finishAppend();
        }
        else {
            // Internal data structure.
            VTCode nextCode = 0;
            std::unordered_map<VTVal, VTCode> recodeDict;

            // Recode the data.
            res->prepareAppend();
            for (size_t r = 0; r < numRowsArg; ++r) {
                const VTVal argVal = arg->get(r, 0);

                auto it = recodeDict.find(argVal);
                if (it != recodeDict.end())
                    res->append(r, 0, it->second);
                else {
                    recodeDict.emplace(argVal, nextCode);
                    res->append(r, 0, nextCode++);
                }
            }
            res->finishAppend();

            // Allocate output for the decoding dictionary.
            if (dict == nullptr)
                dict = DataObjectFactory::create<DenseMatrix<VTVal>>(recodeDict.size(), 1, false);

            // Store decoding dictionary.
            // recodeDict is unordered so we cannot use append here
            for (auto it = recodeDict.begin(); it != recodeDict.end(); ++it)
                dict->set(it->second, 0, it->first);
        }
    }
};