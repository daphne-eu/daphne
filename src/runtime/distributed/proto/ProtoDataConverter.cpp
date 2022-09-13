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


// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------


template<typename VT>
void ProtoDataConverter<DenseMatrix<VT>>::convertToProto(const DenseMatrix<VT> *mat,
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
void ProtoDataConverter<DenseMatrix<VT>>::convertToProto(const DenseMatrix<VT> *mat, distributed::Matrix *matProto)
{
    convertToProto(mat, matProto, 0, mat->getNumRows(), 0, mat->getNumCols());
}

template<typename VT>
void ProtoDataConverter<DenseMatrix<VT>>::convertFromProto(const distributed::Matrix &matProto,
                                          DenseMatrix<VT> *mat,
                                          size_t rowBegin,
                                          size_t rowEnd,
                                          size_t colBegin,
                                          size_t colEnd)
{
    auto denseMatProto = matProto.dense_matrix();
    auto cells = getCells(&matProto);
    for (auto r = rowBegin; r < rowEnd; ++r) {
        for (auto c = colBegin; c < colEnd; ++c) {
            auto val = cells.Get((r - rowBegin) * matProto.num_cols() + (c - colBegin));
            mat->set(r, c, val);
        }
    }
}
template<typename VT>
void ProtoDataConverter<DenseMatrix<VT>>::convertFromProto(const distributed::Matrix &matProto, DenseMatrix<VT> *mat)
{
    convertFromProto(matProto, mat, 0, mat->getNumRows(), 0, mat->getNumCols());
}

template<>
google::protobuf::RepeatedField<int64_t> *ProtoDataConverter<DenseMatrix<int64_t>>::getMutableCells(distributed::Matrix *matProto)
{
    return matProto->mutable_dense_matrix()->mutable_cells_i64()->mutable_cells();
}
template<>
google::protobuf::RepeatedField<double> *ProtoDataConverter<DenseMatrix<double>>::getMutableCells(distributed::Matrix *matProto)
{
    return matProto->mutable_dense_matrix()->mutable_cells_f64()->mutable_cells();
}

template<>
const google::protobuf::RepeatedField<int64_t> ProtoDataConverter<DenseMatrix<int64_t>>::getCells(const distributed::Matrix *matProto)
{
    return matProto->dense_matrix().cells_i64().cells();
}
template<>
const google::protobuf::RepeatedField<double> ProtoDataConverter<DenseMatrix<double>>::getCells(const distributed::Matrix *matProto)
{
    return matProto->dense_matrix().cells_f64().cells();
}

// ----------------------------------------------------------------------------
// CSRMatrix
// ----------------------------------------------------------------------------

template<typename VT>
void ProtoDataConverter<CSRMatrix<VT>>::convertToProto(const CSRMatrix<VT> *mat, distributed::Matrix *matProto)
{
    convertToProto(mat, matProto, 0, mat->getNumRows(), 0, mat->getNumCols());
}
template<typename VT>
void ProtoDataConverter<CSRMatrix<VT>>::convertToProto(const CSRMatrix<VT> *mat,
                                        distributed::Matrix *matProto,
                                        size_t rowBegin,
                                        size_t rowEnd,
                                        size_t colBegin,
                                        size_t colEnd)
{
    auto csrMatProto = matProto->mutable_csr_matrix();
    matProto->set_num_rows(rowEnd - rowBegin);
    matProto->set_num_cols(colEnd - colBegin);

    auto *values = getMutableCells(matProto);    
    auto *colIdxsProto = csrMatProto->mutable_colidx()->mutable_cells();
    auto *rowOffsetsProto = csrMatProto->mutable_row_offsets()->mutable_cells();
        
    // Allocate memory only for the values within the given range.
    size_t numNonZeros = 0;
    for (size_t r = rowBegin; r < rowEnd; r++)
        numNonZeros += mat->getNumNonZeros(r);
    values->Reserve(numNonZeros);
    colIdxsProto->Reserve(numNonZeros);

    rowOffsetsProto->Reserve(rowEnd - rowBegin + 1);
    
    rowOffsetsProto->Add(0);
    auto lastOffset = 0;
    for (auto r = rowBegin; r < rowEnd; r++) {
        const size_t rowNumNonZeros = mat->getNumNonZeros(r);
        const size_t * rowColIdxs = mat->getColIdxs(r);
        const VT *rowValues = mat->getValues(r);
        rowOffsetsProto->Add(lastOffset + rowNumNonZeros);
        lastOffset += rowNumNonZeros;
        for(size_t i = 0; i < rowNumNonZeros; i++) {
            values->Add(rowValues[i]);
            colIdxsProto->Add(rowColIdxs[i]);
        }
    }
}
template<typename VT>
void ProtoDataConverter<CSRMatrix<VT>>::convertFromProto(const distributed::Matrix &matProto,
                                          CSRMatrix<VT> *mat,
                                          size_t rowBegin,
                                          size_t rowEnd,
                                          size_t colBegin,
                                          size_t colEnd)
{    
    auto csrMatProto = matProto.csr_matrix();
 
    auto valuesProto = getCells(&matProto);    
    auto colIdxsProto = csrMatProto.colidx().cells();
    auto rowOffsetsProto = csrMatProto.row_offsets().cells();

    assert (rowBegin < rowEnd && "ProtoDataConverter: rowBegin must be lower than rowEnd");
    if (rowBegin == 0)
        mat->getRowOffsets()[0] = 0;
    // Else rowOffset[rowBegin] is already set.

    size_t protoValColIdx = 0;
    size_t protoRow = 0;
    for (size_t r = rowBegin; r < rowEnd; r++) {
        size_t rowNumNonZeros = rowOffsetsProto[protoRow + 1] - rowOffsetsProto[protoRow];
        protoRow++;
        mat->getRowOffsets()[r + 1] = mat->getRowOffsets()[r] + rowNumNonZeros;

        size_t * rowColIdxs = mat->getColIdxs(r);
        VT *rowValues = mat->getValues(r);
        for (size_t i = 0; i < rowNumNonZeros; i++) {
            rowValues[i] = valuesProto[protoValColIdx];
            rowColIdxs[i] = colIdxsProto[protoValColIdx];
            protoValColIdx++;
        }
    }
}
template<typename VT>
void ProtoDataConverter<CSRMatrix<VT>>::convertFromProto(const distributed::Matrix &matProto, CSRMatrix<VT> *mat)
{
    convertFromProto(matProto, mat, 0, mat->getNumRows(), 0, mat->getNumCols());
}

template<>
google::protobuf::RepeatedField<int64_t> *ProtoDataConverter<CSRMatrix<int64_t>>::getMutableCells(distributed::Matrix *matProto)
{
    return matProto->mutable_csr_matrix()->mutable_values_i64()->mutable_cells();
}
template<>
google::protobuf::RepeatedField<double> *ProtoDataConverter<CSRMatrix<double>>::getMutableCells(distributed::Matrix *matProto)
{
    return matProto->mutable_csr_matrix()->mutable_values_f64()->mutable_cells();
}

template<>
const google::protobuf::RepeatedField<int64_t> ProtoDataConverter<CSRMatrix<int64_t>>::getCells(const distributed::Matrix *matProto)
{
    return matProto->csr_matrix().values_i64().cells();
}
template<>
const google::protobuf::RepeatedField<double> ProtoDataConverter<CSRMatrix<double>>::getCells(const distributed::Matrix *matProto)
{
    return matProto->csr_matrix().values_f64().cells();
}



template class ProtoDataConverter<DenseMatrix<double>>;
template class ProtoDataConverter<DenseMatrix<int64_t>>;
template class ProtoDataConverter<CSRMatrix<double>>;
template class ProtoDataConverter<CSRMatrix<int64_t>>;
