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

#include "runtime/local/context/DaphneContext.h"
#include "runtime/local/datastructures/DataObjectFactory.h"
#include "runtime/local/datastructures/DenseMatrix.h"
#include "HostUtils.h"
#include "runtime/local/kernels/Pooling.h"

namespace CUDA::NN::Pooling {

    // introduce generic pooling ops in this namespace
    using ::NN::Pooling::AVG;
    using ::NN::Pooling::MAX;
    
    template<template<typename> class OP, typename DTRes, typename DTArg>
    struct Forward {
        static void apply(DTRes *&res, size_t& res_h, size_t& res_w, const DTArg *data,
                size_t batch_size, size_t num_channels, size_t img_h, size_t img_w,
                size_t pool_h, size_t pool_w, size_t stride_h, size_t stride_w,
                size_t pad_h, size_t pad_w, DCTX(dctx));
    };
}
