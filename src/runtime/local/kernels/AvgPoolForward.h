#include "Pooling.h"

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes, class DTArg> struct AvgPoolForward {
    static void apply(DTRes *&res, size_t &res_h, size_t &res_w,
                      const DTArg *data, const size_t batch_size,
                      const size_t num_channels, const size_t img_h,
                      const size_t img_w, const size_t pool_h,
                      const size_t pool_w, const size_t stride_h,
                      const size_t stride_w, const size_t pad_h,
                      const size_t pad_w, DCTX(dctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes, class DTArg>
void avgPoolForward(DTRes *&res, size_t &res_h, size_t &res_w,
                    const DTArg *data, const size_t batch_size,
                    const size_t num_channels, const size_t img_h,
                    const size_t img_w, const size_t pool_h,
                    const size_t pool_w, const size_t stride_h,
                    const size_t stride_w, const size_t pad_h,
                    const size_t pad_w, DCTX(dctx)) {
    AvgPoolForward<DTRes, DTArg>::apply(
        res, res_h, res_w, data, batch_size, num_channels, img_h, img_w, pool_h,
        pool_w, stride_h, stride_w, pad_h, pad_w, dctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template <typename VTRes, typename VTArg>
struct AvgPoolForward<DenseMatrix<VTRes>, DenseMatrix<VTArg>> {
    static void apply(DenseMatrix<VTRes> *&res, size_t &res_h, size_t &res_w,
                      const DenseMatrix<VTArg> *data, const size_t batch_size,
                      const size_t num_channels, const size_t img_h,
                      const size_t img_w, const size_t pool_h,
                      const size_t pool_w, const size_t stride_h,
                      const size_t stride_w, const size_t pad_h,
                      const size_t pad_w, DCTX(dctx)) {
        NN::Pooling::Forward<NN::Pooling::AVG, DenseMatrix<VTRes>,
                             DenseMatrix<VTArg>>::apply(res, res_h, res_w, data,
                                                        batch_size,
                                                        num_channels, img_h,
                                                        img_w, pool_h, pool_w,
                                                        stride_h, stride_w,
                                                        pad_h, pad_w, dctx);
    }
};