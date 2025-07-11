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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_MAP_H
#define SRC_RUNTIME_LOCAL_KERNELS_MAP_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Matrix.h>

#include <algorithm>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes, class DTArg> struct Map {
    // We could have a more specialized function pointer here i.e.
    // (DTRes::VT)(*func)(DTArg::VT). The problem is that this is currently not
    // supported by kernels.json.
    static void apply(DTRes *&res, const DTArg *arg, void *func, const int64_t axis, const bool udfReturnsScalar,
                      DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes, class DTArg>
void map(DTRes *&res, const DTArg *arg, void *func, const int64_t axis, const bool udfReturnsScalar, DCTX(ctx)) {
    Map<DTRes, DTArg>::apply(res, arg, func, axis, udfReturnsScalar, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template <typename VTRes, typename VTArg> struct Map<DenseMatrix<VTRes>, DenseMatrix<VTArg>> {
    static void apply(DenseMatrix<VTRes> *&res, const DenseMatrix<VTArg> *arg, void *func, const int64_t axis,
                      const bool udfReturnsScalar, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        VTRes *valuesRes = nullptr;

        if (axis != 0 && axis != 1) { // element-wise
            res = DataObjectFactory::create<DenseMatrix<VTRes>>(numRows, numCols, false);
            valuesRes = res->getValues();
        }

        auto udfElem = reinterpret_cast<VTRes (*)(VTArg)>(func);
        auto udfMatMat = reinterpret_cast<DenseMatrix<VTRes> *(*)(DenseMatrix<VTArg> *)>(func);
        auto udfMatElem = reinterpret_cast<VTRes (*)(DenseMatrix<VTArg> *)>(func);

        if (axis == 1) { // column-wise
            size_t resNumRows = 1;
            for (size_t c = 0; c < numCols; c++) {
                // Extract current column
                DenseMatrix<VTArg> *currentCol =
                    DataObjectFactory::create<DenseMatrix<VTArg>>(arg, 0, numRows, c, c + 1);
                if (!udfReturnsScalar) { // Default: Matrix -> Matrix
                    // Apply UDF and check shape of resulting column
                    const DenseMatrix<VTRes> *resCol = udfMatMat(currentCol);
                    if (resCol == nullptr || resCol->getNumCols() != 1)
                        throw std::runtime_error("UDF return value should be a single column!");
                    // Set result matrix size in first iteration
                    if (c == 0) {
                        resNumRows = resCol->getNumRows();
                        res = DataObjectFactory::create<DenseMatrix<VTRes>>(resNumRows, numCols, false);
                    }
                    // Set result row-wise
                    valuesRes = res->getValues();
                    const VTRes *valuesResCol = resCol->getValues();
                    for (size_t r = 0; r < resNumRows; r++) {
                        valuesRes[c] = valuesResCol[0];
                        valuesResCol += resCol->getRowSkip();
                        valuesRes += res->getRowSkip();
                    }
                } else { // Matrix -> Scalar
                    // Set result matrix size in first iteration
                    if (c == 0) {
                        res = DataObjectFactory::create<DenseMatrix<VTRes>>(resNumRows, numCols, false);
                        valuesRes = res->getValues();
                    }
                    valuesRes[c] = udfMatElem(currentCol);
                }
                // TODO(#XXX): Call to destroy leads to res becoming a view -> segfault in print
                // DataObjectFactory::destroy(currentCol);
            }
        } else { // row-wise or element-wise
            const VTArg *valuesArg = arg->getValues();
            size_t resNumCols = 1;
            for (size_t r = 0; r < numRows; r++) {
                if (axis == 0) {
                    // Extract current row
                    DenseMatrix<VTArg> *currentRow =
                        DataObjectFactory::create<DenseMatrix<VTArg>>(arg, r, r + 1, 0, numCols);
                    if (!udfReturnsScalar) { // Default: Matrix -> Matrix
                        // Apply UDF and check shape of resulting row
                        const DenseMatrix<VTRes> *resRow = udfMatMat(currentRow);
                        if (resRow == nullptr || resRow->getNumRows() != 1)
                            throw std::runtime_error("UDF return value should be a single row!");
                        // Set result matrix size in first iteration
                        if (r == 0) {
                            resNumCols = resRow->getNumCols();
                            res = DataObjectFactory::create<DenseMatrix<VTRes>>(numRows, resNumCols, false);
                            valuesRes = res->getValues();
                        }
                        // Set result by copying values
                        const VTRes *valuesResRow = resRow->getValues();
                        memcpy(valuesRes, valuesResRow, resNumCols * sizeof(VTRes));
                    } else { // Matrix -> Scalar
                        // Set result matrix size in first iteration
                        if (r == 0) {
                            res = DataObjectFactory::create<DenseMatrix<VTRes>>(numRows, resNumCols, false);
                            valuesRes = res->getValues();
                        }
                        valuesRes[0] = udfMatElem(currentRow);
                    }
                    // TODO(#XXX): Call to destroy leads to res becoming a view -> segfault in print
                    // DataObjectFactory::destroy(currentRow);
                } else {
                    for (size_t c = 0; c < numCols; c++)
                        valuesRes[c] = udfElem(valuesArg[c]);
                    valuesArg += arg->getRowSkip();
                }
                valuesRes += res->getRowSkip();
            }
        }
    }
};

// ----------------------------------------------------------------------------
// Matrix
// ----------------------------------------------------------------------------

template <typename VTRes, typename VTArg> struct Map<Matrix<VTRes>, Matrix<VTArg>> {
    static void apply(Matrix<VTRes> *&res, const Matrix<VTArg> *arg, void *func, const int64_t axis,
                      const bool udfReturnsScalar, DCTX(ctx)) {
        // const size_t numRows = arg->getNumRows();
        // const size_t numCols = arg->getNumCols();

        // if (axis != 0 && axis != 1) // element-wise
        //     res = DataObjectFactory::create<DenseMatrix<VTRes>>(numRows, numCols, false);

        // auto udfElem = reinterpret_cast<VTRes (*)(VTArg)>(func);
        // auto udfMatMat = reinterpret_cast<Matrix<VTRes>* (*)(Matrix<VTArg>*)>(func);

        // res->prepareAppend();
        // if (axis == 1) { // column-wise
        //     size_t resNumRows = 1;
        //     // Extract each column, apply udf and set result row-wise
        //     for (size_t c = 0; c < numCols; c++) {
        //         Matrix<VTArg> *currentCol = DataObjectFactory::create<DenseMatrix<VTArg>>(dynamic_cast<const
        //         DenseMatrix<VTArg>*>(arg), 0, numRows, c, c + 1); // TODO(#520) how to extract row/col? const
        //         Matrix<VTRes> *resCol = udfMatMat(currentCol); if (c == 0) {
        //             // Set result matrix size in first iteration
        //             resNumRows = resCol->getNumRows();
        //             res = DataObjectFactory::create<DenseMatrix<VTRes>>(resNumRows, numCols, false);
        //         }
        //         for (size_t r = 0; r < resNumRows; r++)
        //             res->append(r, c, udfElem(resCol->get(r, 0)));
        //         DataObjectFactory::destroy(currentCol);
        //     }
        // } else { // row-wise or element-wise
        //     size_t resNumCols = 1;
        //     for (size_t r = 0; r < numRows; r++) {
        //         if (axis == 0) {
        //             // Extract each row, apply udf and set result column-wise
        //             Matrix<VTArg> *currentRow = DataObjectFactory::create<DenseMatrix<VTArg>>(dynamic_cast<const
        //             DenseMatrix<VTArg>*>(arg), r, r + 1, 0, numCols); const Matrix<VTRes> *resRow =
        //             udfMatMat(currentRow); if (r == 0) {
        //                 // Set result matrix size in first iteration
        //                 resNumCols = resRow->getNumCols();
        //                 res = DataObjectFactory::create<DenseMatrix<VTRes>>(numRows, resNumCols, false);
        //             }
        //             for (size_t c = 0; c < resNumCols; ++c)
        //                 res->append(r, c, resRow->get(0, c));
        //             DataObjectFactory::destroy(currentRow);
        //         } else {
        //             for (size_t c = 0; c < numCols; ++c)
        //                 res->append(r, c, udfElem(arg->get(r, c)));
        //         }
        //     }
        // }
        // res->finishAppend();
    }
};

#endif // SRC_RUNTIME_LOCAL_KERNELS_MAP_H
