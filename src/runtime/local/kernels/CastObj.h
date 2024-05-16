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
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/ValueTypeCode.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTArg>
struct CastObj {
    static void apply(DTRes *& res, const DTArg * arg, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

/**
 * @brief Performs a cast of the given data object to another type.
 * 
 * @param arg The data object to cast.
 * @return The casted data object.
 */
template<class DTRes, class DTArg>
void castObj(DTRes *& res, const DTArg * arg, DCTX(ctx)) {
    CastObj<DTRes, DTArg>::apply(res, arg, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- Frame
// ----------------------------------------------------------------------------

template<typename VTRes>
class CastObj<DenseMatrix<VTRes>, Frame> {
    
    /**
     * @brief Casts the values of the input column at index `c` and stores the
     * casted values to column `c` in the output matrix.
     * @param res The output matrix.
     * @param argFrm The input frame.
     * @param c The position of the column to cast.
     */
    template<typename VTArg>
    static void castCol(DenseMatrix<VTRes> * res, const Frame * argFrm, size_t c) {
        const size_t numRows = argFrm->getNumRows();
        const DenseMatrix<VTArg> * argCol = argFrm->getColumn<VTArg>(c);
        for(size_t r = 0; r < numRows; r++)
            res->set(r, c, static_cast<VTRes>(argCol->get(r, 0)));
    }
    
public:
    static void apply(DenseMatrix<VTRes> *& res, const Frame * arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();
        if(numCols == 1 && arg->getColumnType(0) == ValueTypeUtils::codeFor<VTRes>) {
            // The input frame has a single column of the result's value type.
            // Zero-cost cast from frame to matrix.
            // TODO This case could even be used for (un)signed integers of the
            // same width, involving a reinterpret cast of the pointers.
            // TODO Can we avoid this const_cast?
            res = const_cast<DenseMatrix<VTRes> *>(arg->getColumn<VTRes>(0));
        }
        else {
            // The input frame has multiple columns and/or other value types
            // than the result.
            // Need to change column-major to row-major layout and/or cast the
            // individual values.
            if(res == nullptr)
                res = DataObjectFactory::create<DenseMatrix<VTRes>>(numRows, numCols, false);
            // TODO We could run over the rows in blocks for cache efficiency.
            for(size_t c = 0; c < numCols; c++) {
                // TODO We do not really need all cases.
                // - All pairs of the same type can be handled by a single
                //   copy-the-column helper function.
                // - All pairs of (un)signed integer types of the same width
                //   as well.
                // - Truncating integers to a narrower type does not need to
                //   consider (un)signedness either.
                // - ...
                switch(arg->getColumnType(c)) {
                    // For all value types:
                    case ValueTypeCode::F64: castCol<double>(res, arg, c); break;
                    case ValueTypeCode::F32: castCol<float >(res, arg, c); break;
                    case ValueTypeCode::SI64: castCol<int64_t>(res, arg, c); break;
                    case ValueTypeCode::SI32: castCol<int32_t>(res, arg, c); break;
                    case ValueTypeCode::SI8 : castCol<int8_t >(res, arg, c); break;
                    case ValueTypeCode::UI64: castCol<uint64_t>(res, arg, c); break;
                    case ValueTypeCode::UI32: castCol<uint32_t>(res, arg, c); break;
                    case ValueTypeCode::UI8 : castCol<uint8_t >(res, arg, c); break;
                    default: throw std::runtime_error("CastObj::apply: unknown value type code");
                }
            }
        }
    }
};

// ----------------------------------------------------------------------------
//  Frame <- DenseMatrix
// ----------------------------------------------------------------------------

template<typename VTArg>
class CastObj<Frame, DenseMatrix<VTArg>> {

public:
    static void apply(Frame *& res, const DenseMatrix<VTArg> * arg, DCTX(ctx)) {
        const size_t numCols = arg->getNumCols();
        const size_t numRows = arg->getNumRows();
        std::vector<Structure *> cols;
        if (numCols == 1 && arg->getRowSkip() == 1) {
            // The input matrix has a single column and is not a view into a
            // column range of another matrix, so it can be reused as the
            // column matrix of the output frame.
            // Cheap/Low-cost cast from dense matrix to frame.
            cols.push_back(const_cast<DenseMatrix<VTArg> *>(arg));
        }
        else {
            // The input matrix has multiple columns.
            // Need to change row-major to column-major layout and 
            // split matrix into single column matrices.
            for(size_t c = 0; c < numCols; c++) {
                auto * colMatrix = DataObjectFactory::create<DenseMatrix<VTArg>>(numRows, 1, false);
                for(size_t r = 0; r < numRows; r++)
                    colMatrix->set(r, 0, arg->get(r, c));
                cols.push_back(colMatrix);
            }   
        }
        res = DataObjectFactory::create<Frame>(cols, nullptr);
    }
};

// ----------------------------------------------------------------------------
//  DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template<typename VTRes, typename VTArg>
class CastObj<DenseMatrix<VTRes>, DenseMatrix<VTArg>> {

public:
    static void apply(DenseMatrix<VTRes> *& res, const DenseMatrix<VTArg> * arg, DCTX(ctx)) {
        const size_t numCols = arg->getNumCols();
        const size_t numRows = arg->getNumRows();

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTRes>>(numRows, numCols, false);
        
        auto resVals = res->getValues();
        auto argVals = arg->getValues();

        if(arg->getRowSkip() == numCols && res->getRowSkip() == numCols)
            // Since DenseMatrix implementation is backed by 
            // a single dense array of values, we can simply 
            // perform cast in one loop over that array.
            for(size_t idx = 0; idx < numCols*numRows; idx++)
                resVals[idx] = static_cast<VTRes>(argVals[idx]);
        else
            // res and arg might be views into a larger DenseMatrix.
            for(size_t r = 0; r < numRows; r++) {
                for(size_t c = 0; c < numCols; c++)
                    resVals[c] = static_cast<VTRes>(argVals[c]);
                resVals += res->getRowSkip();
                argVals += arg->getRowSkip();
            }
    }
};


// ----------------------------------------------------------------------------
//  DenseMatrix <- CSRMatrix
// ----------------------------------------------------------------------------

template<typename VT>
class CastObj<DenseMatrix<VT>, CSRMatrix<VT>> {

public:
    static void apply(DenseMatrix<VT> *& res, const CSRMatrix<VT> * arg, DCTX(ctx)) {
        const size_t numCols = arg->getNumCols();
        const size_t numRows = arg->getNumRows();
        
        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false);
        
        // TODO This could be done more efficiently by avoiding the get()/set()
        // calls (use append() or direct access to the underlying arrays).
        VT temp;
        for (size_t r=0; r<numRows; r++){
            for (size_t c=0; c<numCols; c++){
                temp=arg->get(r,c);
                res->set(r,c,temp);
            }
        }
    }
};


// ----------------------------------------------------------------------------
//  CSRMatrix  <- DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
class CastObj<CSRMatrix<VT>, DenseMatrix<VT>> {

public:
    static void apply(CSRMatrix<VT> *& res, const DenseMatrix<VT> * arg, DCTX(ctx)) {
        const size_t numCols = arg->getNumCols();
        const size_t numRows = arg->getNumRows();
        size_t numNonZeros=0;
        VT temp;

        for (size_t r=0; r<numRows; r++){
            for (size_t c=0; c<numCols; c++){
                temp=arg->get(r,c);
                if (temp!=0)
                    numNonZeros++;
            }
        }
        
        if(res == nullptr)
            res = DataObjectFactory::create<CSRMatrix<VT>>(numRows, numCols, numNonZeros, true);
        
        // TODO This could be done more efficiently by avoiding the get()/set()
        // calls (use append() or direct access to the underlying arrays, then
        // we could even avoid initializing the output CSRMatrix).
        for (size_t r=0; r<numRows; r++){
            for (size_t c=0; c<numCols; c++){
                temp=arg->get(r,c);
                res->set(r,c,temp);
            }
        }
    }
};

// ----------------------------------------------------------------------------
//  CSRMatrix  <- DenseMatrix
// ----------------------------------------------------------------------------

template<typename VTres, typename VTarg>
class CastObj<CSRMatrix<VTres>, CSRMatrix<VTarg>> {

public:
    static void apply(CSRMatrix<VTres> *& res, const CSRMatrix<VTarg> * arg, DCTX(ctx)) {
        if(res == nullptr)
            res = DataObjectFactory::create<CSRMatrix<VTres>>(arg->getNumCols(), arg->getNumRows(), arg->getNumNonZeros(), true);

        auto res_val = res->getValues();
        auto res_cidx = res->getColIdxs();
        auto res_roff = res->getRowOffsets();

        auto arg_val = arg->getValues();
        auto arg_cidx = arg->getColIdxs();
        auto arg_roff = arg->getRowOffsets();

        for (size_t nz=0; nz<arg->getNumNonZeros(); nz++){
            res_val[nz] = static_cast<VTres>(arg_val[nz]);
            res_cidx[nz] = arg_cidx[nz];
        }

        for(size_t r = 0; r < arg->getNumRows()+1; r++) {
            res_roff[r] = arg_roff[r];
        }
    }
};

// ----------------------------------------------------------------------------
//  Matrix <- Matrix
// ----------------------------------------------------------------------------

template<typename VTRes, typename VTArg>
class CastObj<Matrix<VTRes>, Matrix<VTArg>> {

public:
    static void apply(Matrix<VTRes> *& res, const Matrix<VTArg> * arg, DCTX(ctx)) {
        const size_t numCols = arg->getNumCols();
        const size_t numRows = arg->getNumRows();

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTRes>>(numRows, numCols, false);

        res->prepareAppend();
        for (size_t r = 0; r < numRows; ++r)
            for (size_t c = 0; c < numCols; ++c)
                res->append(r, c, static_cast<VTRes>(arg->get(r, c)));
        res->finishAppend();
    }
};