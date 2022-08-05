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

#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/Like.h>
#include <tags.h>

#include <catch.hpp>

#include <vector>

void checkLikeMat(const DenseMatrix<const char*>* arg, const size_t colIdx, const char* pattern, const DenseMatrix<const char*>* exp) {
    DenseMatrix<const char*> * res = nullptr;
    like(res, arg, colIdx, pattern, nullptr);
    for(size_t val = 0; val < exp->getNumItems(); val++)
        CHECK(strcmp(res->getValues()[val], exp->getValues()[val]) == 0);
}

TEMPLATE_PRODUCT_TEST_CASE("like", TAG_KERNELS, (DenseMatrix), const char*) {
    using DTArg = TestType;
    auto m1 = genGivenVals<DTArg>(5, {"12", "Prasliker", "51",  
                                    "54", "Trampoler", "77",
                                    "22", "Maxasminlike", "43",  
                                    "98", "Phonyname", "85",
                                    "123", "Maraslimilikes","222"});
    const char* pattern = "%as%like_";
    auto exp = genGivenVals<DTArg>(2, {"12", "Prasliker", "51",   
                                    "123", "Maraslimilikes","222", });
    checkLikeMat(m1, 1, pattern, exp);
    DataObjectFactory::destroy(m1, exp);
}
