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

#ifndef SRC_RUNTIME_DISTRIBUTED_UTILS_PROTODATACONVERTER_H
#define SRC_RUNTIME_DISTRIBUTED_UTILS_PROTODATACONVERTER_H

#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/distributed/proto/worker.pb.h>
#include <runtime/distributed/proto/worker.grpc.pb.h>

template<class DT>
class ProtoDataConverter
{ };

// Partial Specilized class for DenseMatrix 
template<typename VT>
class ProtoDataConverter<DenseMatrix<VT>>
{
private:
    static const google::protobuf::RepeatedField<VT> getCells(const distributed::Matrix *matProto);
    static google::protobuf::RepeatedField<VT> *getMutableCells(distributed::Matrix *matProto);
public:
    static void convertToProto(const DenseMatrix<VT> *mat, distributed::Matrix *matProto);
    static void convertToProto(const DenseMatrix<VT> *mat,
                               distributed::Matrix *matProto,
                               size_t rowBegin,
                               size_t rowEnd,
                               size_t colBegin,
                               size_t colEnd);
    static void convertFromProto(const distributed::Matrix &matProto, DenseMatrix<VT> *mat);
    static void convertFromProto(const distributed::Matrix &matProto,
                                 DenseMatrix<VT> *mat,
                                 size_t rowBegin,
                                 size_t rowEnd,
                                 size_t colBegin,
                                 size_t colEnd);
};

/* Cover const DenseMatrix case with the same implementation */
template<typename VT>
class ProtoDataConverter<const DenseMatrix<VT>> : public ProtoDataConverter<DenseMatrix<VT>>
{ /* Nothing to implement here */ };


// Partial Specilized class for CSRMatrix 
template<typename VT>
class ProtoDataConverter<CSRMatrix<VT>>
{
private:
    static const google::protobuf::RepeatedField<VT> getCells(const distributed::Matrix *matProto);
    static google::protobuf::RepeatedField<VT> *getMutableCells(distributed::Matrix *matProto);
public:
    // Overloaded functions for Sparse Matrices
    static void convertToProto(const CSRMatrix<VT> *mat, distributed::Matrix *matProto);
    static void convertToProto(const CSRMatrix<VT> *mat,
                               distributed::Matrix *matProto,
                               size_t rowBegin,
                               size_t rowEnd,
                               size_t colBegin,
                               size_t colEnd);
    static void convertFromProto(const distributed::Matrix &matProto, CSRMatrix<VT> *mat);
    static void convertFromProto(const distributed::Matrix &matProto,
                                 CSRMatrix<VT> *mat,
                                 size_t rowBegin,
                                 size_t rowEnd,
                                 size_t colBegin,
                                 size_t colEnd);
};

/* Cover const CSRMatrix case with the same implementation */
template<typename VT>
class ProtoDataConverter<const CSRMatrix<VT>> : public ProtoDataConverter<CSRMatrix<VT>>
{ /* TODO */ };

#endif //SRC_RUNTIME_DISTRIBUTED_UTILS_PROTODATACONVERTER_H