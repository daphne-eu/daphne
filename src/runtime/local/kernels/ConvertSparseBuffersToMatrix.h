#pragma once

#include "runtime/local/context/DaphneContext.h"
#include "runtime/local/datastructures/CSRMatrix.h"

#include <iostream>

template <typename T>
inline void convertSparseBuffersToMatrix(CSRMatrix<T> *&result, size_t valuesBase, size_t valuesOffset,
                                         size_t valuesSize, size_t valuesStride, size_t colIdxsBase,
                                         size_t colIdxsOffset, size_t colIdxsSize, size_t colIdxsStride,
                                         size_t rowOffsetsBase, size_t rowOffsetsOffset, size_t rowOffsetsSize,
                                         size_t rowOffsetsStride, size_t numRows, size_t numCols, DCTX(ctx)) {
    std::cout << "[convertSparseBuffersToMatrix][convertSparseBuffersToMatrix][convertSparseBuffersToMatrix]["
                 "convertSparseBuffersToMatrix]\n";
    auto noOpValDeleter = [](T *) {};
    auto noOpIdxDeleter = [](size_t *) {};

    T *valsPtr = reinterpret_cast<T *>(valuesBase) + valuesOffset * valuesStride;
    size_t *colPtr = reinterpret_cast<size_t *>(colIdxsBase) + colIdxsOffset * colIdxsStride;
    size_t *rowPtr = reinterpret_cast<size_t *>(rowOffsetsBase) + rowOffsetsOffset * rowOffsetsStride;

    const size_t rowOffsetsLen = rowOffsetsSize;
    const size_t actualRows = rowOffsetsLen > 0 ? rowOffsetsLen - 1 : 0;
    const size_t rows = std::min(numRows, actualRows);
    const size_t maxNNZ = valuesSize;

    std::cout << "[convertSparseBuffersToMatrix] rows=" << rows << " requestedRows=" << numRows << " cols=" << numCols
              << " valuesLen=" << valuesSize << " rowOffsetsLen=" << rowOffsetsLen << " maxNNZ=" << maxNNZ
              << " valuesPtr=" << static_cast<const void *>(valsPtr) << " colPtr=" << static_cast<const void *>(colPtr)
              << " rowPtr=" << static_cast<const void *>(rowPtr) << std::endl;

    result = DataObjectFactory::create<CSRMatrix<T>>(valsPtr, colPtr, rowPtr, rows, numCols, maxNNZ, noOpValDeleter,
                                                     noOpIdxDeleter);
}
