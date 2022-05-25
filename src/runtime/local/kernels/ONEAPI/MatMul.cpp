/*
 * Copyright 2022 The DAPHNE Consortium
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

#include "MatMul.h"
#include <CL/sycl.hpp>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/<version>/include/dpc_common.hpp
#include "dpc_common.hpp"

// Based on OneAPI sample code
// https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/DPC%2B%2B/DenseLinearAlgebra/matrix_mul

namespace ONEAPI {
    template<typename VT>
    void MatMul<DenseMatrix<VT>, DenseMatrix<VT>, DenseMatrix<VT>>::apply(DenseMatrix<VT> *&res, const DenseMatrix<VT> *lhs,
            const DenseMatrix<VT> *rhs, DCTX(dctx)) {
        std::cout << "\n--- ToDo: ONEAPI MatMul ---\n" << std::endl;

        const size_t nr1 = lhs->getNumRows();
        const size_t nc1 = lhs->getNumCols();
        const size_t nr2 = rhs->getNumRows();
        const size_t nc2 = rhs->getNumCols();

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(nr1, nc2, false);
    
        try {
            sycl::queue q(sycl::default_selector{}, dpc_common::exception_handler);
    
            std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
 
            sycl::buffer a_buf(lhs->getValues(), sycl::range(nr1, nc1));
            sycl::buffer b_buf(rhs->getValues(), sycl::range(nr2, nc2));
            sycl::buffer c_buf(res->getValues(), sycl::range(nr1, nc2));
    
            // Submit command group to queue to multiply matrices: c = a * b
            q.submit([&](auto &h) {
                // Read from a and b, write to c
                sycl::accessor a(a_buf, h, sycl::read_only);
                sycl::accessor b(b_buf, h, sycl::read_only);
                sycl::accessor c(c_buf, h, sycl::write_only);
        
                int width_a = a_buf.get_range()[1];
        
                // Execute kernel.
                h.parallel_for(sycl::range(nr1, nc2), [=](auto index) {
                    // Get global position in Y direction.
                    int row = index[0];
                    // Get global position in X direction.
                    int col = index[1];
            
                    VT sum = 0.0f;
            
                    // Compute the result of one element of c
                    for (int i = 0; i < width_a; i++) {
                        sum += a[row][i] * b[i][col];
                    }
            
                    c[index] = sum;
                });
            });
        }
        catch (sycl::exception const &e) {
            std::cout << "An exception is caught while multiplying matrices.\n";
            std::terminate();
        }
    }

    template struct MatMul<DenseMatrix<float>, DenseMatrix<float>, DenseMatrix<float>>;
    template struct MatMul<DenseMatrix<double>, DenseMatrix<double>, DenseMatrix<double>>;
}