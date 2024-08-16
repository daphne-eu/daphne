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

#include <limits>
#include <random>
#include <type_traits>

#include <cstddef>
#include <cstdint>

namespace NN::Pooling {

    template<typename VT>
    struct AVG {
        static inline VT run(VT initial_value, const VT *in, uint32_t start, uint32_t length, VT plen) {
            VT ret = 0;
            auto end = start + length;
            for (auto i = start; i < end; ++i)
                ret += in[i];
            return ret * plen + initial_value;
        }

        static inline VT getNeutralElement() { return 0; }
        static inline bool isMAX() { return false; }
    };

    template<typename VT>
    struct MAX {
        static inline VT
        run(VT initial_value, const VT *in, uint32_t start, uint32_t length, __attribute__((unused)) VT plen) {
            VT ret = initial_value;
            auto end = start + length;
            for (auto i = start; i < end; ++i)
                ret = std::max(ret, in[i]);
            return ret;
        }

        static inline VT getNeutralElement() { return std::numeric_limits<VT>::max(); }
        static inline bool isMAX() { return true; }
    };

    template<template<typename> class OP, typename DTRes, typename DTArg>
    struct Forward {
        static void apply(DTRes *&res, size_t& res_h, size_t& res_w,
                          const DTArg *data, const size_t batch_size, const size_t num_channels, const size_t img_h, const size_t img_w,
                          const size_t pool_h, const size_t pool_w, const size_t stride_h, const size_t stride_w, const size_t pad_h,
                          const size_t pad_w, DCTX(dctx));
    };
}
