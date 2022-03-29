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

#include "CSRMatrix.h"

template<>
google::protobuf::RepeatedField<int64_t> *CSRMatrix<int64_t>::getMutableCells(distributed::Matrix *matProto) const
{
    return matProto->mutable_csr_matrix()->mutable_values_i64()->mutable_cells();
}
template<>
google::protobuf::RepeatedField<double> *CSRMatrix<double>::getMutableCells(distributed::Matrix *matProto) const
{
    return matProto->mutable_csr_matrix()->mutable_values_f64()->mutable_cells();
}
template<typename ValueType>
google::protobuf::RepeatedField<ValueType> *CSRMatrix<ValueType>::getMutableCells(distributed::Matrix *matProto) const
{
    assert("not implemented yet for this type");
}

template<typename ValueType>
void CSRMatrix<ValueType>::convertToProto(distributed::Matrix *matProto) const
{
    convertToProto(matProto, 0, numRows, 0, numCols);
}

template<typename ValueType>
void CSRMatrix<ValueType>::convertToProto(distributed::Matrix *matProto,
                        size_t rowBegin,
                        size_t rowEnd,
                        size_t colBegin,
                        size_t colEnd) const
{        
    auto csrMatProto = matProto->mutable_csr_matrix();
    matProto->set_num_rows(rowEnd - rowBegin);
    matProto->set_num_cols(colEnd - colBegin);

    auto *values = getMutableCells(matProto);
    auto *colIdxsProto = csrMatProto->mutable_colidx()->mutable_cells();
    auto *rowOffsetsProto = csrMatProto->mutable_row_offsets()->mutable_cells();
    
    values->Reserve(getNumNonZeros());
    colIdxsProto->Reserve(maxNumNonZeros);
    rowOffsetsProto->Reserve(numRows + 1);
    
    rowOffsetsProto->Add(0);
    auto lastOffset = 0;
    for (size_t r = rowBegin; r < rowEnd; r++) {
        const size_t rowNumNonZeros = getNumNonZeros(r);
        const size_t * rowColIdxs = getColIdxs(r);
        const ValueType *rowValues = getValues(r);
        rowOffsetsProto->Add(lastOffset + rowNumNonZeros);
        lastOffset += rowNumNonZeros;
        for(size_t i = 0; i < rowNumNonZeros; i++) {
            values->Add(rowValues[i]);
            colIdxsProto->Add(rowColIdxs[i]);
        }

    }
}


template<>
const google::protobuf::RepeatedField<int64_t> CSRMatrix<int64_t>::getCells(const distributed::Matrix *matProto)
{
    return matProto->csr_matrix().values_i64().cells();
}
template<>
const google::protobuf::RepeatedField<double> CSRMatrix<double>::getCells(const distributed::Matrix *matProto)
{
    return matProto->csr_matrix().values_f64().cells();
}
template<typename ValueType>
const google::protobuf::RepeatedField<ValueType> CSRMatrix<ValueType>::getCells(const distributed::Matrix *matProto)
{
    assert("not implemented yet for this type");
}

template<typename ValueType>
void CSRMatrix<ValueType>::convertFromProto (const distributed::Matrix &matProto)
{
    convertFromProto(matProto, 0, numRows, 0, numCols);
}

template<typename ValueType>
void CSRMatrix<ValueType>::convertFromProto (const distributed::Matrix &matProto,
                        size_t rowBegin,
                        size_t rowEnd,
                        size_t colBegin,
                        size_t colEnd)
{
    auto csrMatProto = matProto.csr_matrix();
    auto valuesProto = getCells(&matProto);    
    auto colIdxsProto = csrMatProto.colidx().cells();
    auto rowOffsetsProto = csrMatProto.row_offsets().cells();
    size_t protoIndexing = 0;
    for (size_t r = rowBegin; r < rowEnd; r++) {
        rowOffsets.get()[r] = rowOffsetsProto[r];
        size_t rowNumNonZeros = rowOffsetsProto[r + 1] - rowOffsetsProto[r];
        size_t * rowColIdxs = getColIdxs(r);
        ValueType *rowValues = getValues(r);
        for (size_t i = 0; i < rowNumNonZeros; i++) {
            rowValues[i] = valuesProto[protoIndexing];
            rowColIdxs[i] = colIdxsProto[protoIndexing];
            protoIndexing++;
        }
    }
    rowOffsets.get()[rowEnd] = rowOffsetsProto[rowEnd];        
}
// explicitly instantiate to satisfy linker
template class CSRMatrix<double>;
template class CSRMatrix<float>;
template class CSRMatrix<int>;
template class CSRMatrix<long>;
template class CSRMatrix<signed char>;
template class CSRMatrix<unsigned char>;
template class CSRMatrix<unsigned int>;
template class CSRMatrix<unsigned long>;
