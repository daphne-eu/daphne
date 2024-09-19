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

#pragma once

#include "Tasks.h"

template <class DT>
class CompiledPipelineTaskCUDA : public CompiledPipelineTaskBase<DT> {};

template <typename VT>
class CompiledPipelineTaskCUDA<DenseMatrix<VT>>
    : public CompiledPipelineTaskBase<DenseMatrix<VT>> {
    std::mutex &_resLock;
    DenseMatrix<VT> ***_res;
    using CompiledPipelineTaskBase<DenseMatrix<VT>>::_data;

  public:
    CompiledPipelineTaskCUDA(CompiledPipelineTaskData<DenseMatrix<VT>> data,
                             std::mutex &resLock, DenseMatrix<VT> ***res)
        : CompiledPipelineTaskBase<DenseMatrix<VT>>(data), _resLock(resLock),
          _res(res) {}

    void execute(uint32_t fid, uint32_t batchSize) override;

    uint64_t getTaskSize() override;

  private:
    void accumulateOutputs(std::vector<DenseMatrix<VT> *> &localResults,
                           std::vector<DenseMatrix<VT> *> &localAddRes,
                           uint64_t rowStart, uint64_t rowEnd);
};
