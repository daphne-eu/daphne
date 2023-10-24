/*
 * Copyright 2023 The DAPHNE Consortium
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

#include <cblas.h>
#include <Eigen/Dense>
#include <spdlog/spdlog.h>

// ****************************************************************************
// DOT
// ****************************************************************************
template<typename T>
T launch_dot(const int32_t n, const T* x, const int32_t incx, const T* y, const int32_t incy);

template<>
float launch_dot(const int32_t n, const float* x, const int32_t incx, const float* y, const int32_t incy) {
    return cblas_sdot(n, x, incx, y, incy);
}

template<>
double launch_dot(const int32_t n, const double* x, const int32_t incx, const double* y, const int32_t incy) {
    return cblas_ddot(n, x, incx, y, incy);
}

template<>
int32_t launch_dot(const int32_t n, const int32_t* x, const int32_t incx, const int32_t* y, const int32_t incy) {
    Eigen::Map<const Eigen::Vector<int32_t, Eigen::Dynamic>, Eigen::Unaligned> eigenX(x, n);
    Eigen::Map<const Eigen::Vector<int32_t, Eigen::Dynamic>, Eigen::Unaligned> eigenY(y, n);
    return eigenX.dot(eigenY);
}

template<>
int64_t launch_dot(const int32_t n, const int64_t* x, const int32_t incx, const int64_t* y, const int32_t incy) {
    Eigen::Map<const Eigen::Vector<int64_t, Eigen::Dynamic>, Eigen::Unaligned> eigenX(x, n);
    Eigen::Map<const Eigen::Vector<int64_t, Eigen::Dynamic>, Eigen::Unaligned> eigenY(y, n);
    return eigenX.dot(eigenY);
}


// ****************************************************************************
// GEMV
// ****************************************************************************
template<typename T>
void launch_gemv(bool transa, bool transb, size_t m, size_t n, const T alpha, const T* A, const int32_t lda,
                 const T* x, const int32_t incx, const T beta, T* y, const int32_t incy);

template<>
void launch_gemv(bool transa, bool transb, size_t m, size_t n, const float alpha, const float* A, const int32_t lda,
                 const float* x, const int32_t incx, const float beta, float* y, const int32_t incy) {
        cblas_sgemv(CblasRowMajor, transa ? CblasTrans : CblasNoTrans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

template<>
void launch_gemv(bool transa, bool transb, size_t m, size_t n, const double alpha, const double* A, const int32_t lda,
                 const double* x, const int32_t incx, const double beta, double* y, const int32_t incy) {
    cblas_dgemv(CblasRowMajor, transa ? CblasTrans : CblasNoTrans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

template<>
void launch_gemv(bool transa, bool transb, size_t m, size_t n, const int32_t alpha, const int32_t* A, const int32_t lda,
                 const int32_t* x, const int32_t incx, const int32_t beta, int32_t* y, const int32_t incy) {
    if(transa) {
        auto e_A = Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic>::Map(A, m, n,
                Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(1, n)).transpose();

        Eigen::Map<const Eigen::Vector<int32_t, Eigen::Dynamic>> e_x(x, m);
        Eigen::Map<Eigen::Vector<int32_t, Eigen::Dynamic>> e_y(y, n);
        e_y.noalias() = e_A * e_x;

    }
    else {
        auto e_A = Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic>::Map(A, m, n,
                Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(1, n));

        Eigen::Map<const Eigen::Vector<int32_t, Eigen::Dynamic>> e_x(x, n);
        Eigen::Map<Eigen::Vector<int32_t, Eigen::Dynamic>> e_y(y, m);
        e_y.noalias() = e_A * e_x;

    }
}

template<>
void launch_gemv(bool transa, bool transb, size_t m, size_t n, const int64_t alpha, const int64_t* A, const int32_t lda,
                 const int64_t* x, const int32_t incx, const int64_t beta, int64_t* y, const int32_t incy) {
    if(transa) {
        auto e_A = Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic>::Map(A, m, n,
                Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(1, n)).transpose();

        Eigen::Map<const Eigen::Vector<int64_t, Eigen::Dynamic>> e_x(x, m);
        Eigen::Map<Eigen::Vector<int64_t, Eigen::Dynamic>> e_y(y, n);
        e_y.noalias() = e_A * e_x;

    }
    else {
        auto e_A = Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic>::Map(A, m, n,
                Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(1, n));

        Eigen::Map<const Eigen::Vector<int64_t, Eigen::Dynamic>> e_x(x, n);
        Eigen::Map<Eigen::Vector<int64_t, Eigen::Dynamic>> e_y(y, m);
        e_y.noalias() = e_A * e_x;

    }
}

// ****************************************************************************
// GEMM
// ****************************************************************************
template<typename T>
void launch_gemm(bool transa, bool transb, const int32_t m, const int32_t n, const int32_t k, const T alpha, const T* A,
        int32_t lda, const T* B, int32_t ldb, const T beta, T *C, int32_t ldc);

template<>
[[maybe_unused]] void launch_gemm<float>(bool transa, bool transb, const int32_t m, const int32_t n, const int32_t k,
        const float alpha, const float* A, const int32_t lda, const float* B, const int32_t ldb, const float beta,
        float *C, const int32_t ldc) {
    cblas_sgemm(CblasRowMajor, transa ? CblasTrans : CblasNoTrans, transb ? CblasTrans : CblasNoTrans,
            m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<>
[[maybe_unused]] void launch_gemm<double>(bool transa, bool transb, const int32_t m, const int32_t n, const int32_t k,
        const double alpha, const double* A, const int32_t lda, const double* B, const int32_t ldb, const double beta,
        double *C, const int32_t ldc) {
    cblas_dgemm(CblasRowMajor, transa ? CblasTrans : CblasNoTrans, transb ? CblasTrans : CblasNoTrans,
            m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<>
[[maybe_unused]] void launch_gemm<int32_t>(bool transa, bool transb, const int32_t m, const int32_t n, const int32_t k,
        const int32_t alpha, const int32_t* A, int32_t lda, const int32_t* B, int32_t ldb, const int32_t beta, int32_t *C,
        int32_t ldc) {

    if(transa && transb) {
        auto eigenA = Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic>::Map(A, k, m,
                                                                                  Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(
                                                                                          1, m)).transpose();
        auto eigenB = Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic>::Map(B, n, k,
                                                                                  Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(
                                                                                          1, k)).transpose();
        auto eigenC = Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic>::Map(C, n, m,
                                                                                  Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(
                                                                                          1, m));
        eigenC.noalias() = eigenA * eigenB;

    }
    else if (transa) {
        auto eigenA = Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic>::Map(A, k, m,
                                                                                  Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(
                                                                                          1, m)).transpose();
        auto eigenB = Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic>::Map(B, n, k,
                                                                                  Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(
                                                                                          1, k));
        auto eigenC = Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic>::Map(C, n, m,
                                                                                  Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(
                                                                                          1, m));
        eigenC.noalias() = eigenA * eigenB;
    }
    else if (transb) {
        auto eigenA = Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic>::Map(A, m, k,
                                                                                  Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(
                                                                                          1, k));
        auto eigenB = Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic>::Map(B, n, k,
                                                                                  Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(
                                                                                          1, k)).transpose();
        auto eigenC = Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic>::Map(C, m, n,
                                                                                  Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(
                                                                                          1, n));
        eigenC.noalias() = eigenA * eigenB;
    }
    else {
        auto eigenA = Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic>::Map(A, m, k,
                                                                                  Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(1, k));
        auto eigenB = Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic>::Map(B, k, n,
                                                                                  Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(1, n));
        auto eigenC = Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic>::Map(C, m, n,
                                                                                  Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(1, n));
        eigenC.noalias() = eigenA * eigenB;

    }
}

template<>
[[maybe_unused]] void launch_gemm<int64_t>(bool transa, bool transb, const int32_t m, const int32_t n, const int32_t k,
        const int64_t alpha, const int64_t* A, int32_t lda, const int64_t* B, int32_t ldb, const int64_t beta, int64_t *C,
        int32_t ldc) {
    if(transa && transb) {
        auto eigenA = Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic>::Map(A, k, m,
                                                                                  Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(
                                                                                          1, m)).transpose();
        auto eigenB = Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic>::Map(B, n, k,
                                                                                  Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(
                                                                                          1, k)).transpose();
        auto eigenC = Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic>::Map(C, n, m,
                                                                                  Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(
                                                                                          1, m));
        eigenC.noalias() = eigenA * eigenB;

    }
    else if (transa) {
        auto eigenA = Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic>::Map(A, k, m,
                                                                                  Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(
                                                                                          1, m)).transpose();
        auto eigenB = Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic>::Map(B, n, k,
                                                                                  Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(
                                                                                          1, k));
        auto eigenC = Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic>::Map(C, n, m,
                                                                                  Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(
                                                                                          1, m));
        eigenC.noalias() = eigenA * eigenB;
    }
    else if (transb) {
        auto eigenA = Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic>::Map(A, m, k,
                                                                                  Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(
                                                                                          1, k));
        auto eigenB = Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic>::Map(B, n, k,
                                                                                  Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(
                                                                                          1, k)).transpose();
        auto eigenC = Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic>::Map(C, m, n,
                                                                                  Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(
                                                                                          1, n));
        eigenC.noalias() = eigenA * eigenB;
    }
    else {
        auto eigenA = Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic>::Map(A, m, k,
                                                                                  Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(1, k));
        auto eigenB = Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic>::Map(B, k, n,
                                                                                  Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(1, n));
        auto eigenC = Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic>::Map(C, m, n,
                                                                                  Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(1, n));
        eigenC.noalias() = eigenA * eigenB;

    }
}

template<typename VT>
void MatMul<DenseMatrix<VT>, DenseMatrix<VT>, DenseMatrix<VT>>::apply(DenseMatrix<VT> *&res, const DenseMatrix<VT> *lhs,
        const DenseMatrix<VT> *rhs, bool transa, bool transb, DCTX(dctx)) {
    const auto nr1 = static_cast<int>(transa ? lhs->getNumCols() : lhs->getNumRows());
    const auto nc1 = static_cast<int>(transa ? lhs->getNumRows() : lhs->getNumCols());
    const auto nr2 = static_cast<int>(transb ? rhs->getNumCols() : rhs->getNumRows());
    const auto nc2 = static_cast<int>(transb ? rhs->getNumRows() : rhs->getNumCols());
    assert((nc1 == nr2) && "#cols of lhs and #rows of rhs must be the same");
    const VT alpha = 1.0f;
    const VT beta = 0.0f;
    if(res == nullptr)
        res = DataObjectFactory::create<DenseMatrix<VT>>(nr1, nc2, false);

    // adding BLAS nomenclature - should be optimized away by the compiler ;-)
    auto m = nr1;
    auto n = nc2;
    auto k = nr2;
    auto lda = lhs->getRowSkip();
    auto ldb = rhs->getRowSkip();
    auto ldc = res->getRowSkip();

    const auto A = lhs->getValues();
    const auto B = rhs->getValues();
    auto C = res->getValues();

    if(nr1 == 1 && nc2 == 1) {// Vector-Vector
        dctx->logger->debug("launch_dot<{}>(a[{}x{}], b[{}x{}])", typeid(alpha).name(), m, k, k, n);
        res->set(0, 0, launch_dot(nc1, A, 1, B, rhs->isView() ? (transb ? 1 : rhs->getRowSkip()) : 1));
    }
    else if(nc2 == 1) {      // Matrix-Vector
        dctx->logger->debug("launch_gemv<{}>(A[{},{}], x[{}])", typeid(alpha).name(), m, k, k);
        launch_gemv<VT>(transa, transb, lhs->getNumRows(), lhs->getNumCols(), alpha, A, lda, B, 1, beta, C, 1);
    }
    else { // Matrix-Matrix
        dctx->logger->debug("launch_gemm<{}>(C[{}x{}], A[{},{}], B[{}x{}], transA:{}, transB:{})",
                typeid(alpha).name(), m, n, m, k, k, n, transa, transb);
        launch_gemm<VT>(transa, transb, nr1, nc2, nc1, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}


// explicit instantiations to satisfy linker
template struct MatMul<DenseMatrix<float>, DenseMatrix<float>, DenseMatrix<float>>;
template struct MatMul<DenseMatrix<double>, DenseMatrix<double>, DenseMatrix<double>>;
template struct MatMul<DenseMatrix<int32_t>, DenseMatrix<int32_t>, DenseMatrix<int32_t>>;
template struct MatMul<DenseMatrix<int64_t>, DenseMatrix<int64_t>, DenseMatrix<int64_t>>;
