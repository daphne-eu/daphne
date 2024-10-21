#ifndef WRITE_MM_H
#define WRITE_MM_H

#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/io/MMFile.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <type_traits>
#include <vector>

template <class DTArg> struct WriteMM {
    static void apply(const DTArg *arg, const char *filename) = delete;
};

template <class DTArg> void writeMM(const DTArg *arg, const char *filename);

template <typename VT> struct WriteMM<DenseMatrix<VT>> {
    static void apply(const DenseMatrix<VT> *arg, const char *filename);

  private:
    static bool isSymmetric(const DenseMatrix<VT> *arg);
    static bool isSkewSymmetric(const DenseMatrix<VT> *arg);
};

template <typename VT> struct WriteMM<CSRMatrix<VT>> {
    static void apply(const CSRMatrix<VT> *arg, const char *filename);
};

#endif // WRITE_MM_H
