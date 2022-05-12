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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_GROUP_H
#define SRC_RUNTIME_LOCAL_KERNELS_GROUP_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/ValueTypeCode.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>
#include <runtime/local/kernels/Order.h>
#include <runtime/local/kernels/ExtractCol.h>
#include <util/DeduceType.h>
#include <ir/daphneir/Daphne.h>

#include <iterator>
#include <vector>

using mlir::daphne::GroupEnumAttr;
using mlir::daphne::GroupEnum;

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes>
struct Group {
    static void apply(DTRes *& res, const DTRes * arg,const char ** keyCols, size_t numKeyCols,
        const char ** aggCols, size_t numAggCols, GroupEnumAttr * aggFuncs, size_t numAggFuncs, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes>
void group(DTRes *& res, const DTRes * arg, const char ** keyCols, size_t numKeyCols,
        const char ** aggCols, size_t numAggCols, GroupEnumAttr * aggFuncs, size_t numAggFuncs, DCTX(ctx)) {
    Group<DTRes>::apply(res, arg, keyCols, numKeyCols, aggCols, numAggCols, aggFuncs, numAggFuncs, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// Frame <- Frame
// ----------------------------------------------------------------------------

// returns the result of the aggregation function aggFunc over the (contiguous) memory between the begin and end pointer 
template<typename VTRes, typename VTArg>
VTRes aggregate (const GroupEnum & aggFunc, const VTArg * begin, const VTArg* end) { 
    switch(aggFunc) {
        case GroupEnum::COUNT: return end-begin; break; // TODO: Do we need to check for Null elements here?
        case GroupEnum::SUM: return std::accumulate(begin, end, (VTRes) 0); break;
        case GroupEnum::MIN: return *std::min_element(begin, end); break;
        case GroupEnum::MAX: return *std::max_element(begin, end); break; 
        case GroupEnum::AVG: return std::accumulate(begin, end, (double) 0)/(double) (end-begin); break;
        default : return *begin; break;
    }
}

// struct which calls the aggregate() function (specified via aggFunc) on each duplicate group in the groups vector and on
// all implied single groups for a sepcified column (colIdx) of the argument frame (arg) and stores the result in the
// specified column (colIdx) of the result frame (res)
template<typename VTRes, typename VTArg>
struct ColumnGroupAgg {
    static void apply(Frame * res, const Frame * arg, size_t colIdx, std::vector<std::pair<size_t, size_t>> * groups, GroupEnum aggFunc, DCTX(ctx)) {
        VTRes * valuesRes = res->getColumn<VTRes>(colIdx)->getValues();
        const VTArg * valuesArg = arg->getColumn<VTArg>(colIdx)->getValues();
        size_t rowRes = 0;
        size_t numRows = arg->getNumRows();

        // case for no duplicates
        if (groups == nullptr || groups->empty()) {
            for(size_t r = 0; r < numRows; r++) 
                valuesRes[rowRes++] = aggregate<VTRes,VTArg>(aggFunc, valuesArg + r, valuesArg + r + 1);
            return;
        }
        
        for(size_t r = 0; r < groups->front().first; r++)
            valuesRes[rowRes++] = aggregate<VTRes,VTArg>(aggFunc, valuesArg + r, valuesArg + r + 1);
        for(auto it = groups->begin(); it != groups->end(); ++it) {
            valuesRes[rowRes++] = aggregate<VTRes,VTArg>(aggFunc, valuesArg + it->first, valuesArg + it->second);
            for(size_t r = it->second; r < (std::next(it) != groups->end() ? std::next(it)->first : it->second); r++){
                valuesRes[rowRes++] = aggregate<VTRes,VTArg>(aggFunc, valuesArg + r, valuesArg + r + 1);
            } 
        }
        for(size_t r = groups->back().second; r < numRows; r++) 
            valuesRes[rowRes++] = aggregate<VTRes,VTArg>(aggFunc, valuesArg + r, valuesArg + r + 1);
    }
};

template <> struct Group<Frame> {
    static void apply(Frame *& res, const Frame * arg, const char ** keyCols, size_t numKeyCols,
        const char ** aggCols, size_t numAggCols, GroupEnumAttr * aggFuncs, size_t numAggFuncs, DCTX(ctx)) {
        size_t numRows = arg->getNumRows();
        size_t numCols = numKeyCols + numAggCols;
        size_t numRowsRes = numRows;
        if (arg == nullptr || keyCols == nullptr || numKeyCols == 0 || aggCols == nullptr || numAggCols == 0 || aggFuncs == nullptr || numAggFuncs == 0) {
            throw std::runtime_error("group-kernel called with invalid arguments");
        }

        // convert labels to indices
        auto idxs = std::shared_ptr<size_t[]>(new size_t[numCols]);
        bool * ascending = new bool[numKeyCols];
        for (size_t i = 0; i < numKeyCols; i++) {
            idxs[i] = arg->getColumnIdx(keyCols[i]);
            ascending[i] = true;
        }   
        for (size_t i = numKeyCols; i < numCols; i++) {
            idxs[i] = arg->getColumnIdx(aggCols[i-numKeyCols]);
        }
        
        // reduce frame columns to keyCols and numAggCols (without copying values or the idx array) and reorder them accordingly 
        Frame* reduced{};
        auto sel = DataObjectFactory::create<DenseMatrix<size_t>>(numCols, 1, idxs);
        extractCol(reduced, arg, sel, ctx);
        DataObjectFactory::destroy(sel);
    
        std::iota(idxs.get(), idxs.get()+numCols, 0);
        auto groups = new std::vector<std::pair<size_t, size_t>>;
        Frame* ordered{};     

        // order frame rows by groups and get the group vector 
        order(ordered, reduced, idxs.get(), numKeyCols, ascending, numKeyCols, false, ctx, groups);
        delete [] ascending;
        DataObjectFactory::destroy(reduced);
        size_t inGroups = 0;
        for (auto & group : *groups){
            inGroups += group.second-group.first;
        }  
        numRowsRes -= inGroups-groups->size();

        // create the result frame
        std::string * labels = new std::string[numCols];
        ValueTypeCode * schema = new ValueTypeCode[numCols];

        for (size_t i = 0; i < numKeyCols; i++) {
            labels[i] = keyCols[i];
            schema[i] = ordered->getColumnType(idxs[i]);
        }
        for (size_t i = numKeyCols; i < numCols; i++) {
            labels[i] = mlir::daphne::stringifyGroupEnum(aggFuncs[i-numKeyCols].getValue()).str() + "(" +  aggCols[i-numKeyCols] + ")";
            switch(aggFuncs[i-numKeyCols].getValue()) {
                case GroupEnum::COUNT: schema[i] = ValueTypeCode::UI64; break;
                case GroupEnum::SUM: schema[i] = ordered->getColumnType(idxs[i]); break;
                case GroupEnum::MIN: schema[i] = ordered->getColumnType(idxs[i]); break;
                case GroupEnum::MAX: schema[i] = ordered->getColumnType(idxs[i]); break;
                case GroupEnum::AVG: schema[i] = ValueTypeCode::F64; break;
            }
        } 
        
        res = DataObjectFactory::create<Frame>(numRowsRes, numCols, schema, labels, false);
        delete [] labels;
        delete [] schema;

        // copying key columns and column-wise group aggregation
        for (size_t i = 0; i < numCols; i++) {
            DeduceValueTypeAndExecute<ColumnGroupAgg>::apply(res->getSchema()[i], ordered->getSchema()[i], res, ordered, i, groups, (i < numKeyCols) ? (GroupEnum) 0 : aggFuncs[i-numKeyCols].getValue(), ctx);
        }        
        delete groups;
        DataObjectFactory::destroy(ordered);
   }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_GROUP_H