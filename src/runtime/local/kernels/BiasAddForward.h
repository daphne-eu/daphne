#pragma once

#include <runtime/local/context/DaphneContext.h>

#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <limits>
#include <random>
#include <type_traits>

#include <cstddef>
#include <cstdint>

#include <cmath>
#include <iostream>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes, class DTArg> struct BiasAddForward {
    static void apply(DTRes *&res, const DTArg *input, const DTArg *bias,
                      DCTX(dctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes, class DTArg>
void biasAddForward(DTRes *&res, const DTArg *input, const DTArg *bias,
                    DCTX(dctx)) {
    BiasAddForward<DTRes, DTArg>::apply(res, input, bias, dctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template <typename VTRes, typename VTArg>
struct BiasAddForward<DenseMatrix<VTRes>, DenseMatrix<VTArg>> {
    static void apply(DenseMatrix<VTRes> *&res, const DenseMatrix<VTArg> *input,
                      const DenseMatrix<VTArg> *bias, DCTX(dctx)) {
        auto start = 0;
        auto stop = input->getNumRows();
        auto C = bias->getNumRows();
        auto CHW = input->getNumCols();
        auto HW = CHW / C;

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTArg>>(
                input->getNumRows(), CHW, true);

        for (uint32_t i = start; i < stop; i++)
            for (uint32_t c = 0; c < C; c++)
                for (uint32_t j = 0; j < HW; j++)
                    res->getValues()[i * CHW + c * HW + j] =
                        input->getValues()[i * CHW + c * HW + j] +
                        bias->getValues()[c];
    }
};