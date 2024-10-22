#pragma once

#include <cstddef>
#include <cstdint>

static uint32_t getPQ(uint32_t img_extent, uint32_t filter_extent, uint32_t pad_extent, uint32_t stride_extent) {
    uint32_t padded_image_extent = img_extent + 2 * pad_extent;
    return (padded_image_extent - filter_extent) / stride_extent + 1;
}

template <typename VT>
static inline void Padding(VT *padded_input, const VT *input, size_t pad_h, size_t pad_w, size_t img_w, size_t img_h,
                           uint32_t off) {
    auto padded_w = img_w + 2 * pad_w;
    for (uint32_t i = 0; i < img_h * img_w; i++)
        padded_input[i] = 0;

    auto start = pad_h * padded_w + pad_w;
    for (uint32_t i = 0, j = 0; i < img_h; i++)
        for (uint32_t k = 0; k < img_w; k++, j++)
            padded_input[start + i * padded_w + k] = input[off + j];
}

template <typename VT>
static inline void CleanPaddingAndSave(VT *res, const VT *dPooling_padded, size_t pad_h, size_t pad_w, size_t img_w,
                                       size_t img_h, uint32_t off_input) {
    auto start = pad_h * (img_w + 2 * pad_w) + pad_w;
    auto off_res = off_input;
    for (uint32_t i = 0, off_padded = 0; i < img_h; i++)
        for (uint32_t j = 0; j < img_w; j++, off_res++) {
            off_padded = start + i * (img_w + 2 * pad_w) + j;
            res[off_res] = dPooling_padded[off_padded];
        }
}