#pragma once

#include <cstdint>
#include <cstddef>

uint32_t getPQ(uint32_t img_extent, uint32_t filter_extent, uint32_t pad_extent, uint32_t stride_extent) 
{
    uint32_t padded_image_extent = img_extent + 2 * pad_extent;
    return (padded_image_extent - filter_extent) / stride_extent + 1;
}

template <typename VT>
static inline void
GetPaddedData(const VT *data, VT *padded_data, VT *selected_data,
              size_t pad_w, size_t pad_h,
              size_t img_w, size_t img_h, size_t padded_img_w, uint32_t off)
{
    uint32_t j = 0;
    uint32_t k = 0;
    uint32_t padded_index = 0;
    uint32_t data_index = 0;
    for (j = 0; j < img_h * img_w; j++)
        selected_data[j] = data[off + j];

    for (j = 0; j < (pad_h * padded_img_w); j++, padded_index++)
        padded_data[padded_index] = 0;
    for (j = 0; j < img_h; j++)
    {
        for (k = 0; k < pad_w; k++, padded_index++)
            padded_data[padded_index] = 0;
        for (k = 0; k < img_w; k++, data_index++, padded_index++)
            padded_data[padded_index] = selected_data[data_index];
        for (k = 0; k < pad_w; k++, padded_index++)
            padded_data[padded_index] = 0;
    }
    for (j = 0; j < (pad_h * padded_img_w); j++, padded_index++)
        padded_data[padded_index] = 0;
}

template <typename VT>
static inline void
Padding(VT *padded_input, const VT *input, size_t pad_h, size_t pad_w, size_t img_w, size_t img_h, uint32_t off)
{
    auto padded_w = img_w + 2 * pad_w;
    for (uint32_t i = 0; i < img_h * img_w; i++)
        padded_input[i] = 0;
    
    auto start = pad_h * padded_w + pad_w;
    for (uint32_t i = 0, j = 0; i < img_h; i++)
        for (uint32_t k = 0; k < img_w; k++, j++)
            padded_input[start + i * padded_w + k] = input[off + j];
}