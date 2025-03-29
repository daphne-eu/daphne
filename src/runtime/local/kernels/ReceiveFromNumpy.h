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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_RECEIVEFROMNUMPY_H
#define SRC_RUNTIME_LOCAL_KERNELS_RECEIVEFROMNUMPY_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <memory>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes> struct ReceiveFromNumpy {
    static void apply(DTRes *&res, uint32_t upper, uint32_t lower, int64_t rows, int64_t cols, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes>
void receiveFromNumpy(DTRes *&res, uint32_t upper, int32_t lower, int64_t rows, int64_t cols, DCTX(ctx)) {
    ReceiveFromNumpy<DTRes>::apply(res, upper, lower, rows, cols, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

// TODO Should we make this a central utility?
template <typename VT> struct NoOpDeleter {
    void operator()(VT *p) {
        // Don't delete p because the memory comes from numpy.
    }
};

template <typename VT> struct ReceiveFromNumpy<DenseMatrix<VT>> {
    static void apply(DenseMatrix<VT> *&res, uint32_t upper, uint32_t lower, int64_t rows, int64_t cols, DCTX(ctx)) {
        std::cerr << "ReceiveFromNumpy: Extracting data from shared memory" << std::endl;
        res = DataObjectFactory::create<DenseMatrix<VT>>(
            rows, cols, std::shared_ptr<VT[]>((VT *)(((uint64_t)upper << 32) | lower), NoOpDeleter<VT>()));
    }
};

template <>
struct ReceiveFromNumpy<DenseMatrix<std::string>> {
    static void apply(DenseMatrix<std::string> *&res, 
                    uint32_t upper, 
                    uint32_t lower, 
                    int64_t rows, 
                    int64_t cols, 
                    DCTX(ctx)) {
        try {
            // Calculate shared memory address
            char* shared_mem = reinterpret_cast<char*>(
                (static_cast<uint64_t>(upper) << 32) | lower;

            // Create result matrix
            res = DataObjectFactory::create<DenseMatrix<std::string>>(rows, cols, false);
            
            // Parse NULL-delimited strings
            for (int64_t r = 0; r < rows; ++r) {
                for (int64_t c = 0; c < cols; ++c) {
                    const char* str_start = shared_mem;
                    
                    // Find string length (until NULL terminator)
                    size_t len = 0;
                    while (shared_mem[len] != '\0' && len < 256) { // 256 = max expected string length
                        len++;
                    }
                    
                    // Construct string properly
                    res->set(r, c, std::string(str_start, len));
                    
                    // Move to next string (skip NULL terminator)
                    shared_mem += len + 1;
                }
            }
        }
        catch (const std::exception& e) {
            if (res) {
                DataObjectFactory::destroy(res);
                res = nullptr;
            }
            std::cerr << "String transfer error: " << e.what() << std::endl;
            throw;
        }
    }
};

#endif // SRC_RUNTIME_LOCAL_KERNELS_RECEIVEFROMNUMPY_H