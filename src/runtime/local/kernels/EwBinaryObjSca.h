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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_EWBINARYOBJSCA_H
#define SRC_RUNTIME_LOCAL_KERNELS_EWBINARYOBJSCA_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/Matrix.h>
#include <runtime/local/kernels/BinaryOpCode.h>
#include <runtime/local/kernels/EwBinarySca.h>


#include <cassert>
#include <cstddef>
#include <cstring>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTLhs, typename VTRhs>
struct EwBinaryObjSca {
    static void apply(BinaryOpCode opCode, DTRes *& res, const DTLhs * lhs, VTRhs rhs, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTLhs, typename VTRhs>
void ewBinaryObjSca(BinaryOpCode opCode, DTRes *& res, const DTLhs * lhs, VTRhs rhs, DCTX(ctx)) {
    EwBinaryObjSca<DTRes, DTLhs, VTRhs>::apply(opCode, res, lhs, rhs, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix, scalar
// ----------------------------------------------------------------------------

template <typename VT>
struct EwBinaryObjSca<DenseMatrix<VT>, DenseMatrix<VT>, VT> {
    static void apply(BinaryOpCode opCode, DenseMatrix<VT>*& res,
                      const DenseMatrix<VT>* lhs, VT rhs, DCTX(ctx)) {
        const size_t numRows = lhs->getNumRows();
        const size_t numCols = lhs->getNumCols();

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols,
                                                             false);

        const VT* valuesLhs = lhs->getValues();
        VT* valuesRes = res->getValues();

        if (opCode == BinaryOpCode::MOD) {
            for (size_t r = 0; r < numRows; r++) {
                for (size_t c = 0; c < numCols; c++)
                    valuesRes[c] = std::fmod(valuesLhs[c], rhs);
                valuesLhs += lhs->getRowSkip();
                valuesRes += res->getRowSkip();
            }
            return;
        }

        EwBinaryScaFuncPtr<VT, VT, VT> func =
            getEwBinaryScaFuncPtr<VT, VT, VT>(opCode);

        // TODO(phil): also remove func for other BinaryOpCode?
        for (size_t r = 0; r < numRows; r++) {
            for (size_t c = 0; c < numCols; c++)
                valuesRes[c] = func(valuesLhs[c], rhs, ctx);
            valuesLhs += lhs->getRowSkip();
            valuesRes += res->getRowSkip();
        }
    }
};

template <>
struct EwBinaryObjSca<DenseMatrix<uint64_t>, DenseMatrix<uint64_t>, uint64_t> {
    static void apply(BinaryOpCode opCode, DenseMatrix<uint64_t>*& res,
                      const DenseMatrix<uint64_t>* lhs, uint64_t rhs, DCTX(ctx)) {
        const size_t numRows = lhs->getNumRows();
        const size_t numCols = lhs->getNumCols();

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<uint64_t>>(numRows, numCols,
                                                             false);

        const uint64_t* valuesLhs = lhs->getValues();
        uint64_t* valuesRes = res->getValues();

        if (opCode == BinaryOpCode::MOD) {
            if (rhs % 2 == 0) {
                for (size_t r = 0; r < numRows; r++) {
                    for (size_t c = 0; c < numCols; c++)
                        valuesRes[c] = valuesLhs[c] & (rhs - 1);
                    valuesLhs += lhs->getRowSkip();
                    valuesRes += res->getRowSkip();
                }
                return;
            }

            for (size_t r = 0; r < numRows; r++) {
                for (size_t c = 0; c < numCols; c++)
                    valuesRes[c] = std::fmod(valuesLhs[c], rhs);
                valuesLhs += lhs->getRowSkip();
                valuesRes += res->getRowSkip();
            }
            return;
        }

        if (opCode == BinaryOpCode::BITWISE_AND) {
            for (size_t r = 0; r < numRows; r++) {
                for (size_t c = 0; c < numCols; c++)
                    valuesRes[c] = valuesLhs[c] & rhs;
                valuesLhs += lhs->getRowSkip();
                valuesRes += res->getRowSkip();
            }
            return;
        }

        EwBinaryScaFuncPtr<uint64_t, uint64_t, uint64_t> func =
            getEwBinaryScaFuncPtr<uint64_t, uint64_t, uint64_t>(opCode);

        // TODO(phil): also remove func for other BinaryOpCode?
        for (size_t r = 0; r < numRows; r++) {
            for (size_t c = 0; c < numCols; c++)
                valuesRes[c] = func(valuesLhs[c], rhs, ctx);
            valuesLhs += lhs->getRowSkip();
            valuesRes += res->getRowSkip();
        }
    }
};

template <>
struct EwBinaryObjSca<DenseMatrix<int64_t>, DenseMatrix<int64_t>, int64_t> {
    static void apply(BinaryOpCode opCode, DenseMatrix<int64_t>*& res,
                      const DenseMatrix<int64_t>* lhs, int64_t rhs, DCTX(ctx)) {
        const size_t numRows = lhs->getNumRows();
        const size_t numCols = lhs->getNumCols();

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<int64_t>>(numRows, numCols,
                                                             false);

        const int64_t* valuesLhs = lhs->getValues();
        int64_t* valuesRes = res->getValues();

        if (opCode == BinaryOpCode::MOD) {
            if (rhs % 2 == 0) {
                for (size_t r = 0; r < numRows; r++) {
                    for (size_t c = 0; c < numCols; c++)
                        valuesRes[c] = valuesLhs[c] & (rhs - 1);
                    valuesLhs += lhs->getRowSkip();
                    valuesRes += res->getRowSkip();
                }
                return;
            }

            for (size_t r = 0; r < numRows; r++) {
                for (size_t c = 0; c < numCols; c++)
                    valuesRes[c] = std::fmod(valuesLhs[c], rhs);
                valuesLhs += lhs->getRowSkip();
                valuesRes += res->getRowSkip();
            }
            return;
        }

        EwBinaryScaFuncPtr<int64_t, int64_t, int64_t> func =
            getEwBinaryScaFuncPtr<int64_t, int64_t, int64_t>(opCode);

        // TODO(phil): also remove func for other BinaryOpCode?
        for (size_t r = 0; r < numRows; r++) {
            for (size_t c = 0; c < numCols; c++)
                valuesRes[c] = func(valuesLhs[c], rhs, ctx);
            valuesLhs += lhs->getRowSkip();
            valuesRes += res->getRowSkip();
        }
    }
};

// ----------------------------------------------------------------------------
// Matrix <- Matrix, scalar
// ----------------------------------------------------------------------------

template<typename VT>
struct EwBinaryObjSca<Matrix<VT>, Matrix<VT>, VT> {
    static void apply(BinaryOpCode opCode, Matrix<VT> *& res, const Matrix<VT> * lhs, VT rhs, DCTX(ctx)) {
        const size_t numRows = lhs->getNumRows();
        const size_t numCols = lhs->getNumCols();
        
        // TODO Choose matrix implementation depending on expected number of non-zeros.
        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false);
        
        EwBinaryScaFuncPtr<VT, VT, VT> func = getEwBinaryScaFuncPtr<VT, VT, VT>(opCode);
        
        res->prepareAppend();
        for(size_t r = 0; r < numRows; r++)
            for(size_t c = 0; c < numCols; c++)
                res->append(r, c) = func(lhs->get(r, c), rhs);
        res->finishAppend();
    }
};


// ----------------------------------------------------------------------------
// Frame <- Frame, scalar
// ----------------------------------------------------------------------------

template<typename VT>
void ewBinaryFrameColSca(BinaryOpCode opCode, Frame *& res, const Frame * lhs, VT rhs, size_t c, DCTX(ctx)) {
    auto * col_res = res->getColumn<VT>(c);
    auto * col_lhs = lhs->getColumn<VT>(c);
    ewBinaryObjSca<DenseMatrix<VT>, DenseMatrix<VT>, VT>(opCode, col_res, col_lhs, rhs, ctx);
}

template<typename VT>
struct EwBinaryObjSca<Frame, Frame, VT> {
    static void apply(BinaryOpCode opCode, Frame *& res, const Frame * lhs, VT rhs, DCTX(ctx)) {
        const size_t numRows = lhs->getNumRows();
        const size_t numCols = lhs->getNumCols();

        if(res == nullptr)
            res = DataObjectFactory::create<Frame>(numRows, numCols, lhs->getSchema(), lhs->getLabels(), false);
        
        for (size_t c = 0; c < numCols; c++) {
            switch(lhs->getColumnType(c)) {
                // For all value types:
                case ValueTypeCode::F64: ewBinaryFrameColSca<double>(opCode, res, lhs, rhs, c, ctx); break;
                case ValueTypeCode::F32: ewBinaryFrameColSca<float>(opCode, res, lhs, rhs, c, ctx); break;
                case ValueTypeCode::SI64: ewBinaryFrameColSca<int64_t>(opCode, res, lhs, rhs, c, ctx); break;
                case ValueTypeCode::SI32: ewBinaryFrameColSca<int32_t>(opCode, res, lhs, rhs, c, ctx); break;
                case ValueTypeCode::SI8 : ewBinaryFrameColSca<int8_t>(opCode, res, lhs, rhs, c, ctx); break;
                case ValueTypeCode::UI64: ewBinaryFrameColSca<uint64_t>(opCode, res, lhs, rhs, c, ctx); break;
                case ValueTypeCode::UI32: ewBinaryFrameColSca<uint32_t>(opCode, res, lhs, rhs, c, ctx); break; 
                case ValueTypeCode::UI8 : ewBinaryFrameColSca<uint8_t>(opCode, res, lhs, rhs, c, ctx); break;
                default: throw std::runtime_error("EwBinaryObjSca::apply: unknown value type code");
            }
        }   
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_EWBINARYOBJSCA_H
