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

// Convenience function
template <class DTArg> void writeMM(const DTArg *arg, const char *filename) { WriteMM<DTArg>::apply(arg, filename); }

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct WriteMM<DenseMatrix<VT>> {
    static void apply(const DenseMatrix<VT> *arg, const char *filename) {
        const char *format = MM_DENSE_STR;
        std::ofstream f(filename);
        if (!f.is_open()) {
            throw std::runtime_error("WriteMM::apply: Cannot open file");
        }

        const char *field;
        if (std::is_integral<VT>::value) {
            field = MM_INT_STR;
        } else if (std::is_floating_point<VT>::value) {
            field = MM_REAL_STR;
        } else {
            throw std::runtime_error("WriteMM::apply: Unsupported data type");
        }

        const char *symmetry = MM_GENERAL_STR;
        if (isSymmetric(arg)) {
            symmetry = MM_SYMM_STR;
        } else if (isSkewSymmetric(arg)) {
            symmetry = MM_SKEW_STR;
        }

        f << MatrixMarketBanner << " matrix " << format << " " << field << " " << symmetry << std::endl;

        size_t rows = arg->getNumRows();
        size_t cols = arg->getNumCols();
        f << rows << " " << cols << std::endl;

        const VT *values = arg->getValues();
        if (!values) {
            throw std::runtime_error("WriteMM::apply: Null pointer for 'values' in DenseMatrix");
        }

        if (strcmp(symmetry, MM_GENERAL_STR) == 0) {
            for (size_t i = 0; i < cols; ++i) {
                for (size_t j = 0; j < rows; ++j) {
                    size_t idx = j * cols + i;
                    if (strcmp(field, MM_REAL_STR) == 0)
                        f << std::scientific << std::setprecision(13) << values[idx] << std::endl;
                    else
                        f << values[idx] << std::endl;
                }
            }
        } else if (strcmp(symmetry, MM_SYMM_STR) == 0) {
            for (size_t i = 0; i < cols; ++i) {
                for (size_t j = i; j < rows; ++j) {
                    size_t idx = j * cols + i;
                    if (strcmp(field, MM_REAL_STR) == 0)
                        f << std::scientific << std::setprecision(13) << values[idx] << std::endl;
                    else
                        f << values[idx] << std::endl;
                }
            }
        } else if (strcmp(symmetry, MM_SKEW_STR) == 0) {
            for (size_t i = 0; i < cols; ++i) {
                for (size_t j = i + 1; j < rows; ++j) {
                    size_t idx = j * cols + i;
                    if (strcmp(field, MM_REAL_STR) == 0)
                        f << std::scientific << std::setprecision(13) << values[idx] << std::endl;
                    else
                        f << values[idx] << std::endl;
                }
            }
        } else {
            throw std::runtime_error("WriteMM::apply: Unsupported symmetry type");
        }
        f.close();
    }

  private:
    static bool isSymmetric(const DenseMatrix<VT> *arg) {
        size_t rows = arg->getNumRows();
        size_t cols = arg->getNumCols();
        if (rows != cols)
            return false;
        const VT *values = arg->getValues();
        for (size_t i = 0; i < cols; ++i) {
            for (size_t j = i + 1; j < rows; ++j) {
                size_t idx1 = j + i * rows;
                size_t idx2 = i + j * rows;
                if (values[idx1] != values[idx2])
                    return false;
            }
        }
        return true;
    }

    static bool isSkewSymmetric(const DenseMatrix<VT> *arg) {
        size_t rows = arg->getNumRows();
        size_t cols = arg->getNumCols();
        if (rows != cols)
            return false;
        const VT *values = arg->getValues();
        for (size_t i = 0; i < rows; ++i) {
            size_t idx_diag = i + i * rows;
            if (values[idx_diag] != 0)
                return false;
        }
        for (size_t i = 0; i < cols; ++i) {
            for (size_t j = i + 1; j < rows; ++j) {
                size_t idx1 = j + i * rows;
                size_t idx2 = i + j * rows;
                if (values[idx1] != -values[idx2])
                    return false;
            }
        }
        return true;
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct WriteMM<CSRMatrix<VT>> {
    static void apply(const CSRMatrix<VT> *arg, const char *filename) {
        const char *format = MM_SPARSE_STR;
        std::ofstream f(filename);
        if (!f.is_open()) {
            throw std::runtime_error("WriteMM::apply: Cannot open file");
        }

        const char *field;
        if (std::is_integral<VT>::value) {
            field = MM_INT_STR;
        } else if (std::is_floating_point<VT>::value) {
            field = MM_REAL_STR;
        } else {
            throw std::runtime_error("WriteMM::apply: Unsupported data type");
        }

        const char *symmetry = MM_GENERAL_STR;

        f << MatrixMarketBanner << " matrix " << format << " " << field << " " << symmetry << std::endl;

        size_t rows = arg->getNumRows();
        size_t cols = arg->getNumCols();
        size_t nnz = countNNZ(arg, symmetry);

        f << rows << " " << cols << " " << nnz << std::endl;

        const size_t *rowOffsets = arg->getRowOffsets();
        const size_t *colIdxs = arg->getColIdxs();
        const VT *values = arg->getValues();

        std::vector<std::vector<std::pair<size_t, VT>>> colEntries(cols);

        for (size_t i = 0; i < rows; ++i) {
            for (size_t idx = rowOffsets[i]; idx < rowOffsets[i + 1]; ++idx) {
                size_t j = colIdxs[idx];
                VT val = values[idx];
                colEntries[j].emplace_back(i, val);
            }
        }

        if (strcmp(symmetry, MM_GENERAL_STR) == 0) {
            for (size_t j = 0; j < cols; ++j) {
                for (const auto &entry : colEntries[j]) {
                    size_t i = entry.first;
                    VT val = entry.second;
                    if (strcmp(field, MM_REAL_STR) == 0) {
                        if (val >= 0) {
                            f << i + 1 << " " << j + 1 << "  " << std::scientific << std::setprecision(13) << val
                              << std::endl;
                        } else {
                            f << i + 1 << " " << j + 1 << " " << std::scientific << std::setprecision(13) << val
                              << std::endl;
                        }
                    } else {
                        f << i + 1 << " " << j + 1 << " " << val << std::endl;
                    }
                }
            }
        } else {
            throw std::runtime_error("WriteMM::apply: Unsupported symmetry type");
        }

        f.close();
    }

  private:
    static size_t countNNZ(const CSRMatrix<VT> *arg, const char *symmetry) {
        size_t nnz = 0;
        size_t rows = arg->getNumRows();
        size_t cols = arg->getNumCols();

        std::vector<std::vector<std::pair<size_t, VT>>> colEntries(cols);

        const size_t *rowOffsets = arg->getRowOffsets();
        const size_t *colIdxs = arg->getColIdxs();
        const VT *values = arg->getValues();

        for (size_t i = 0; i < rows; ++i) {
            for (size_t idx = rowOffsets[i]; idx < rowOffsets[i + 1]; ++idx) {
                size_t j = colIdxs[idx];
                colEntries[j].emplace_back(i, values[idx]);
            }
        }

        if (strcmp(symmetry, MM_GENERAL_STR) == 0) {
            nnz = arg->getNumNonZeros();
        } else {
            throw std::runtime_error("WriteMM::apply: Unsupported symmetry type");
        }

        return nnz;
    }
};

#endif // WRITE_MM_H
