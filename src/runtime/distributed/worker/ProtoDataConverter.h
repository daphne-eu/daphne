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
#include <runtime/distributed/proto/worker.pb.h>
#include <runtime/distributed/proto/worker.grpc.pb.h>

class ProtoDataConverter
{
public:
    static void convertToProto(const DenseMatrix<double> *mat, distributed::Matrix *matProto);
    static void convertToProto(const DenseMatrix<double> *mat,
                               distributed::Matrix *matProto,
                               size_t rowBegin,
                               size_t rowEnd,
                               size_t colBegin,
                               size_t colEnd);
    static void convertFromProto(const distributed::Matrix &matProto, DenseMatrix<double> *mat);
    static void convertFromProto(const distributed::Matrix &matProto,
                                 DenseMatrix<double> *mat,
                                 size_t rowBegin,
                                 size_t rowEnd,
                                 size_t colBegin,
                                 size_t colEnd);
};

#endif //SRC_RUNTIME_DISTRIBUTED_UTILS_PROTODATACONVERTER_H