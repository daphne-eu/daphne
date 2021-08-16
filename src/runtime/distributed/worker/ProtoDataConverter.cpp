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

#include "ProtoDataConverter.h"

void ProtoDataConverter::convertToProto(const DenseMatrix<double> *mat, distributed::Matrix *matProto)
{
    convertToProto(mat, matProto, 0, mat->getNumRows(), 0, mat->getNumCols());
}

void ProtoDataConverter::convertToProto(const DenseMatrix<double> *mat,
                                        distributed::Matrix *matProto,
                                        size_t rowBegin,
                                        size_t rowEnd,
                                        size_t colBegin,
                                        size_t colEnd)
{
    matProto->set_num_rows(rowEnd - rowBegin);
    matProto->set_num_cols(colEnd - colBegin);

    auto *cells = matProto->mutable_cells_f64()->mutable_cells();
    cells->Reserve(mat->getNumRows() * mat->getNumCols());
    for (auto r = rowBegin; r < rowEnd; ++r) {
        for (auto c = colBegin; c < colEnd; ++c) {
            cells->Add(mat->get(r, c));
        }
    }
}

void ProtoDataConverter::convertFromProto(const distributed::Matrix &matProto,
                                          DenseMatrix<double> *mat,
                                          size_t rowBegin,
                                          size_t rowEnd,
                                          size_t colBegin,
                                          size_t colEnd)
{
    auto cells = matProto.cells_f64().cells();
    for (auto r = rowBegin; r < rowEnd; ++r) {
        for (auto c = colBegin; c < colEnd; ++c) {
            auto val = cells.Get((r - rowBegin) * matProto.num_cols() + (c - colBegin));
            mat->set(r, c, val);
        }
    }
}

void ProtoDataConverter::convertFromProto(const distributed::Matrix &matProto, DenseMatrix<double> *mat)
{
    convertFromProto(matProto, mat, 0, mat->getNumRows(), 0, mat->getNumCols());
}