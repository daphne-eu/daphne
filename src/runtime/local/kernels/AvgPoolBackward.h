#include "Padding.h"
#include "Pooling.h"

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes, class DTArg> struct AvgPoolBackward {
    static void apply(DTRes *&res, const DTArg *input, const DTArg *dOut,
                      const size_t batch_size, const size_t num_channels,
                      const size_t img_h, const size_t img_w,
                      const size_t pool_h, const size_t pool_w,
                      const size_t stride_h, const size_t stride_w,
                      const size_t pad_h, const size_t pad_w,
                      DCTX(dctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes, class DTArg>
void avgPoolBackward(DTRes *&res, const DTArg *input, const DTArg *dOut,
                     const size_t batch_size, const size_t num_channels,
                     const size_t img_h, const size_t img_w,
                     const size_t pool_h, const size_t pool_w,
                     const size_t stride_h, const size_t stride_w,
                     const size_t pad_h, const size_t pad_w, DCTX(dctx)) {
    AvgPoolBackward<DTRes, DTArg>::apply(
        res, input, dOut, batch_size, num_channels, img_h, img_w, pool_h,
        pool_w, stride_h, stride_w, pad_h, pad_w, dctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template <typename VTRes, typename VTArg>
struct AvgPoolBackward<DenseMatrix<VTRes>, DenseMatrix<VTArg>> {
    static void apply(DenseMatrix<VTRes> *&res, const DenseMatrix<VTArg> *input,
                      const DenseMatrix<VTArg> *dOut, const size_t batch_size,
                      const size_t num_channels, const size_t img_h,
                      const size_t img_w, const size_t pool_h,
                      const size_t pool_w, const size_t stride_h,
                      const size_t stride_w, const size_t pad_h,
                      const size_t pad_w, DCTX(dctx)) {
        auto HW = img_h * img_w;
        auto C = num_channels;
        auto CHW = C * HW;
        // padded height/width
        auto P = getPQ(img_h, pool_h, pad_h, stride_w);
        auto Q = getPQ(img_w, pool_w, pad_w, stride_h);
        auto CPQ = C * P * Q;
        auto start = 0;
        auto stop = batch_size;
        auto plen = static_cast<VTArg>(1) / static_cast<VTArg>(pool_w * pool_h);

        auto padded_img_h = img_h + 2 * pad_h;
        auto padded_img_w = img_w + 2 * pad_w;
        DenseMatrix<VTArg> *dPooling_padded =
            DataObjectFactory::create<DenseMatrix<VTArg>>(
                1, padded_img_h * padded_img_w, true);

        if (res == nullptr) {
            res = DataObjectFactory::create<DenseMatrix<VTArg>>(batch_size, CHW,
                                                                true);
        }

        for (uint32_t i = start; i < stop; i++)
            for (uint32_t c = 0; c < C; c++) {
                auto off_input = i * CHW + c * HW;
                for (uint32_t p = 0; p < P; p++)
                    for (uint32_t h = 0; h < pool_h; h++)
                        for (uint32_t q = 0; q < Q; q++)
                            for (uint32_t w = 0; w < pool_w; w++) {
                                auto off_padded =
                                    (p * stride_h + h) * padded_img_w +
                                    q * stride_w + w;
                                auto off_output =
                                    i * CPQ + c * P * Q + p * Q + q;
                                dPooling_padded->getValues()[off_padded] =
                                    dPooling_padded->getValues()[off_padded] +
                                    plen * dOut->getValues()[off_output];
                            }
                CleanPaddingAndSave(res->getValues(),
                                    dPooling_padded->getValues(), pad_h, pad_w,
                                    img_w, img_h, off_input);
                for (uint32_t i = 0; i < padded_img_h * padded_img_w; i++)
                    dPooling_padded->getValues()[i] = 0;
            }

        DataObjectFactory::destroy(dPooling_padded);
    }
};