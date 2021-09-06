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

#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/CheckEq.h>

#include <cassert>

#ifdef USE_CUDA
	#include <api/cli/DaphneUserConfig.h>
	#include <runtime/local/kernels/CUDA_BatchNorm.h>
	#include <runtime/local/kernels/CUDA_InitContext.h>
#else
 ToDo: cpu version
	#include <runtime/local/kernels/BatchNorm.h>
#endif

#include <tags.h>
#include <catch.hpp>

template<class DT>
void check(const DT* in, const DT* gamma, const DT* beta, const DT* ema_mean, const DT* ema_var, const DT* exp,
		DaphneContext* dctx)
{
	DT* res = nullptr;
	typename DT::VT epsilon = 1e-5;
#ifdef USE_CUDA
	BatchNorm::ForwardTest_CUDA<DT, DT>::apply(res, in, gamma, beta, ema_mean, ema_var, epsilon, dctx);
#else
	//"ToDo: cpu version
	return;
	BatchNorm::Forward<OP, DT, DT>::apply(res, in, filter, in->getNumRows(), 1, 3, 3);
#endif
	CHECK(Approx(*(res->getValues())).epsilon(epsilon) == *(exp->getValues()));
}

TEMPLATE_PRODUCT_TEST_CASE("batchnorm_fwd", TAG_DNN, (DenseMatrix), (float, double)) {
	using DT = TestType;

	auto dctx = new DaphneContext();
#ifdef USE_CUDA
	initCUDAContext(dctx);
#endif

	auto input = genGivenVals<DT>(1, { -3, -2, -1, 0, 1, 2, 3, 4, 5});
	auto gamma = genGivenVals<DT>(1, { 1 });
	auto beta = genGivenVals<DT>(1, { 0 });
	auto ema_mean = genGivenVals<DT>(1, { 0 });
	auto ema_var = genGivenVals<DT>(1, { 1 });

	auto result = genGivenVals<DT>(1, { -3, -2, -1, 0, 1, 2, 3, 4, 5});

	check(input, gamma, beta, ema_mean, ema_var, result, dctx);

	DataObjectFactory::destroy(input);
	DataObjectFactory::destroy(result);

	delete dctx;
}
