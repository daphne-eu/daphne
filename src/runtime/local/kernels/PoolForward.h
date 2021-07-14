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

#ifndef DAPHNE_PROTOTYPE_POOL_FORWARD_H
#define DAPHNE_PROTOTYPE_POOL_FORWARD_H

#pragma once

#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <random>
#include <type_traits>

#include <cassert>
#include <cstddef>
#include <cstdint>

enum class PoolOpCode {
	AVG,
	MAX
};
// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<PoolOpCode OP, typename DTRes, typename DTArg>
struct PoolForward {
	static void apply(DTRes *& res, const DTArg* data, uint64_t batch_size, uint8_t num_channels, uint32_t img_h,
			uint32_t img_w, uint32_t pool_h, uint32_t pool_w, uint32_t stride_h, uint32_t stride_w, uint32_t pad_h,
			uint32_t pad_w) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

//template<class DTRes, class DTArg>
//void maxPool(DTRes *& res, const DTArg* data, size_t batch_size, size_t num_channels, size_t img_h, size_t img_w,
//		size_t pool_h, size_t pool_w, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w) {
//	Pooling<DTRes, DTArg>::apply_max(res, data, batch_size, num_channels, img_h, img_w, pool_h, pool_w, stride_h,
//			stride_w, pad_h, pad_w);
//}

template <typename DTRes, typename DTArg>
struct PoolForward<PoolOpCode::AVG, DTRes, DTArg> {
	static void apply(DTRes *& res, const DTArg* data, uint64_t batch_size, uint8_t num_channels, uint32_t img_h,
			uint32_t img_w, uint32_t pool_h, uint32_t pool_w, uint32_t stride_h, uint32_t stride_w, uint32_t pad_h,
			uint32_t pad_w) {
		std::cout << "avgPool() has been called" << std::endl;
	}
};

template <typename DTRes, typename DTArg>
struct PoolForward<PoolOpCode::MAX, DTRes, DTArg> {
	static void apply(DTRes *& res, const DTArg* data, uint64_t batch_size, uint8_t num_channels, uint32_t img_h,
			uint32_t img_w, uint32_t pool_h, uint32_t pool_w, uint32_t stride_h, uint32_t stride_w, uint32_t pad_h,
			uint32_t pad_w) {
		std::cout << "maxPool() has been called" << std::endl;
	}
};

#endif //DAPHNE_PROTOTYPE_POOL_FORWARD_H
