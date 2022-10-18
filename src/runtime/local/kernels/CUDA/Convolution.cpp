/*
 * Copyright 2021 The DAPHNE Consortium
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

#include "Convolution.h"
#include <runtime/local/datastructures/AllocationDescriptorCUDA.h>

namespace CUDA::Convolution {
    template<typename DTRes, typename DTArg>
    void Forward<DTRes, DTArg>::apply(DTRes *&res, size_t& res_h, size_t& res_w, const DTArg *data, const DTArg *filter, const DTArg *bias,
            const size_t batch_size, const size_t num_channels, const size_t img_h, const size_t img_w, const size_t filter_h,
            const size_t filter_w, const size_t stride_h, const size_t stride_w, const size_t pad_h, const size_t pad_w, DCTX(dctx))
    {
        const size_t deviceID = 0; //ToDo: multi device support
        auto ctx = CUDAContext::get(dctx, deviceID);
        AllocationDescriptorCUDA alloc_desc(dctx, deviceID);
        
        using VT = typename DTRes::VT;
        auto F = filter->getNumRows(); // num filters
        const VT blend_alpha = 1;
        VT blend_beta = 0;
        const VT* d_input = data->getValues(&alloc_desc);
        const VT* d_filter = filter->getValues(&alloc_desc);

        cudnnConvolutionFwdAlgo_t algo;

        CHECK_CUDNN(cudnnSetTensor4dDescriptor(ctx->src_tensor_desc, ctx->tensor_format, ctx->template getCUDNNDataType<VT>(), batch_size,
                num_channels, img_h, img_w));
        const int tensorDims = 4;
        int tensorOuputDimA[tensorDims];
        const int filterDimA[tensorDims] = {static_cast<int>(F), static_cast<int>(num_channels), static_cast<int>(filter_h),
                static_cast<int>(filter_w)};

        CHECK_CUDNN(cudnnSetFilterNdDescriptor(ctx->filter_desc, ctx->template getCUDNNDataType<VT>(), CUDNN_TENSOR_NCHW, tensorDims, filterDimA));

        const int convDims = 2;
        int padA[convDims] = {static_cast<int>(pad_h), static_cast<int>(pad_w)};
        int filterStrideA[convDims] = { static_cast<int>(stride_h), static_cast<int>(stride_w)};
        int upscaleA[convDims] = {1,1};
        cudnnDataType_t convDataType = ctx->template getCUDNNDataType<VT>();

        // ToDo: Math are done in FP32 when tensor are in FP16.
//        if (ctx->data_type == CUDNN_DATA_HALF) {
//            convDataType = CUDNN_DATA_FLOAT;
//        }

        CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(ctx->conv_desc, convDims, padA, filterStrideA, upscaleA,
                CUDNN_CROSS_CORRELATION, convDataType));

        CHECK_CUDNN(cudnnGetConvolutionNdForwardOutputDim(ctx->conv_desc, ctx->src_tensor_desc, ctx->filter_desc,
                tensorDims, tensorOuputDimA));

        int n = tensorOuputDimA[0]; int c = tensorOuputDimA[1];
        int h = tensorOuputDimA[2]; int w = tensorOuputDimA[3];
        res_h = h;
        res_w = w;
//        size_t out_buf_size = n * c * h * w * sizeOfDataType;
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(ctx->dst_tensor_desc, ctx->tensor_format, ctx->template getCUDNNDataType<VT>(), n, c, h, w));

        if (res == nullptr) {
            res = DataObjectFactory::create<DTRes>(batch_size, c*h*w, false, &alloc_desc);
        }
        
        VT* d_res = res->getValues(&alloc_desc);
        if (ctx->conv_algorithm < 0) {
            int requestedAlgoCount = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
            int returnedAlgoCount = -1;
            cudnnConvolutionFwdAlgoPerf_t results[2 * CUDNN_CONVOLUTION_FWD_ALGO_COUNT];

            CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(ctx->getCUDNNHandle(), ctx->src_tensor_desc, ctx->filter_desc,
                                                             ctx->conv_desc, ctx->dst_tensor_desc, requestedAlgoCount, &returnedAlgoCount, results));
            algo = results[0].algo;
            ctx->conv_algorithm = algo;
        }
        else {
            algo = static_cast<cudnnConvolutionFwdAlgo_t>(ctx->conv_algorithm);
        }

        size_t workspace_sizeInBytes=0;
        void* work_space=nullptr;
        CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(ctx->getCUDNNHandle(), ctx->src_tensor_desc, ctx->filter_desc,
                                                            ctx->conv_desc, ctx->dst_tensor_desc, algo, &workspace_sizeInBytes));

        if (workspace_sizeInBytes!=0) {
            work_space = ctx->getCUDNNWorkspace(workspace_sizeInBytes);
        }

        CHECK_CUDNN(cudnnConvolutionForward(ctx->getCUDNNHandle(), &blend_alpha, ctx->src_tensor_desc, d_input,
                ctx->filter_desc, d_filter, ctx->conv_desc, algo, work_space, workspace_sizeInBytes, &blend_beta,
                ctx->dst_tensor_desc, d_res));

        if(bias) {
            if (bias != filter) {
                const VT *d_bias = bias->getValues(&alloc_desc);
//            CHECK_CUDART(cudaMalloc(reinterpret_cast<void**>(&d_bias), bias->getNumCols() * sizeOfDataType));
//            CHECK_CUDART(cudaMemcpy(d_bias, bias->getValues(), bias->getNumCols() * sizeOfDataType, cudaMemcpyHostToDevice));

                CHECK_CUDNN(cudnnSetTensor4dDescriptor(ctx->src_tensor_desc, ctx->tensor_format,
                                                       ctx->getCUDNNDataType<VT>(), 1,
                                                       c, 1, 1));
                blend_beta = 1;
                CHECK_CUDNN(
                        cudnnAddTensor(ctx->getCUDNNHandle(), &blend_alpha, ctx->src_tensor_desc, d_bias, &blend_beta,
                                       ctx->dst_tensor_desc, d_res));
            }
        }
    }

    template struct Forward<DenseMatrix<float>, DenseMatrix<float>>;
    template struct Forward<DenseMatrix<double>, DenseMatrix<double>>;
}

