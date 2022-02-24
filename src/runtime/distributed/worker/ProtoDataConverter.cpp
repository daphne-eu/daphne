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

template<>
google::protobuf::RepeatedField<int64_t> *ProtoDataConverter<int64_t>::getMutableCells(distributed::Matrix *matProto)
{
    return matProto->mutable_cells_i64()->mutable_cells();
}
template<>
google::protobuf::RepeatedField<double> *ProtoDataConverter<double>::getMutableCells(distributed::Matrix *matProto)
{
    return matProto->mutable_cells_f64()->mutable_cells();
}

template<>
const google::protobuf::RepeatedField<int64_t> ProtoDataConverter<int64_t>::getCells(const distributed::Matrix *matProto)
{
    return matProto->cells_i64().cells();
}
template<>
const google::protobuf::RepeatedField<double> ProtoDataConverter<double>::getCells(const distributed::Matrix *matProto)
{
    return matProto->cells_f64().cells();
}


template<typename VT>
void ProtoDataConverter<VT>::convertToProto(const DenseMatrix<VT> *mat, distributed::Matrix *matProto)
{
    convertToProto(mat, matProto, 0, mat->getNumRows(), 0, mat->getNumCols());
}
template<typename VT>
void ProtoDataConverter<VT>::convertToProto(const DenseMatrix<VT> *mat,
                                        distributed::Matrix *matProto,
                                        size_t rowBegin,
                                        size_t rowEnd,
                                        size_t colBegin,
                                        size_t colEnd)
{
    matProto->set_num_rows(rowEnd - rowBegin);
    matProto->set_num_cols(colEnd - colBegin);

    auto *cells = getMutableCells(matProto);
    cells->Reserve(mat->getNumRows() * mat->getNumCols());
    for (auto r = rowBegin; r < rowEnd; ++r) {
        for (auto c = colBegin; c < colEnd; ++c) {
            cells->Add(mat->get(r, c));
        }
    }
}
template<typename VT>
void ProtoDataConverter<VT>::convertFromProto(const distributed::Matrix &matProto,
                                          DenseMatrix<VT> *mat,
                                          size_t rowBegin,
                                          size_t rowEnd,
                                          size_t colBegin,
                                          size_t colEnd)
{
    //auto cells = matProto.cells_f64().cells();
    auto cells = getCells(&matProto);
    for (auto r = rowBegin; r < rowEnd; ++r) {
        for (auto c = colBegin; c < colEnd; ++c) {
            auto val = cells.Get((r - rowBegin) * matProto.num_cols() + (c - colBegin));
            mat->set(r, c, val);
        }
    }
}
template<typename VT>
void ProtoDataConverter<VT>::convertFromProto(const distributed::Matrix &matProto, DenseMatrix<VT> *mat)
{
    convertFromProto(matProto, mat, 0, mat->getNumRows(), 0, mat->getNumCols());
}

template class ProtoDataConverter<double>;
template class ProtoDataConverter<int64_t>;
