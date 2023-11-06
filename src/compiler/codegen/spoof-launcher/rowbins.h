
#pragma once

#include "Matrix.h"

template<class T>
void rowBins(CCMatrix<T>* mat, uint32_t* bin_sizes, uint32_t* bins, const uint32_t* row_ptrs, size_t num_rows);
