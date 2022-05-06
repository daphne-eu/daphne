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

#include "runtime/local/vectorized/TasksCUDA.h"
#include "runtime/local/kernels/CUDA/EwBinaryMat.h"

template<typename VT>
void CompiledPipelineTaskCUDA<DenseMatrix<VT>>::execute(uint32_t fid, uint32_t batchSize) {
    // local add aggregation to minimize locking
    std::vector<DenseMatrix<VT>*> localAddRes(_data._numOutputs);
    std::vector<DenseMatrix<VT>*> localResults(_data._numOutputs);
    for(uint64_t r = _data._rl ; r < _data._ru ; r += batchSize) {
        //create zero-copy views of inputs/outputs
        uint64_t r2 = std::min(r + batchSize, _data._ru);
        
        auto linputs = this->createFuncInputs(r, r2);
        std::vector<DenseMatrix<VT>**> outputs;
        
        for (auto &lres : localResults) {
            outputs.push_back(&lres);
        }
        //execute function on given data binding (batch size)
        _data._funcs[fid](outputs.data(), linputs.data(), _data._ctx);
        accumulateOutputs(localResults, localAddRes, r, r2);
        
        // cleanup
        for (auto &localResult : localResults) {
            DataObjectFactory::destroy(localResult);
            localResult = nullptr;
        }
    }
    
    for(size_t o = 0; o < _data._numOutputs; ++o) {
        if(_data._combines[o] == VectorCombine::ADD) {
            auto &result = (*_res[o]);
            _resLock.lock();
            if(result == nullptr) {
                result = localAddRes[o];
                _resLock.unlock();
            }
            else {
                CUDA::ewBinaryMat(BinaryOpCode::ADD, result, result, localAddRes[o], _data._ctx);
                _resLock.unlock();
                //cleanup
                DataObjectFactory::destroy(localAddRes[o]);
            }
        }
    }
}

template<typename VT>
void CompiledPipelineTaskCUDA<DenseMatrix<VT>>::accumulateOutputs(std::vector<DenseMatrix<VT>*>& localResults,
        std::vector<DenseMatrix<VT> *> &localAddRes, uint64_t rowStart, uint64_t rowEnd) {
    
    //TODO: in-place computation via better compiled pipelines
    //TODO: multi-return
    for(auto o = 0u ; o < _data._numOutputs ; ++o) {
        auto &result = (*_res[o]);
        switch (_data._combines[o]) {
            case VectorCombine::ROWS: {
                auto bufsize = localResults[o]->bufferSize();
                auto data = result->getValuesCUDA();
                data += result->getRowSkip() * rowStart;
                CHECK_CUDART(cudaMemcpy(data, localResults[o]->getValuesCUDA(), bufsize, cudaMemcpyDeviceToDevice));
                break;
            }
            case VectorCombine::COLS: {
                auto res_base_ptr = result->getValuesCUDA();
                auto lres_data_base_ptr = localResults[o]->getValuesCUDA();
                auto rlen = rowEnd - rowStart;
                auto slice = result->slice(0, this->_data._outRows[o], rowStart, rowEnd);
                for(auto i = 0u; i < slice->getNumRows(); ++i) {
                    auto data_src = lres_data_base_ptr + localResults[o]->getRowSkip() * i;
                    auto data_dst = res_base_ptr + result->getRowSkip() * i + rowStart;
//                    auto data_dst = res_base_ptr;
                    CHECK_CUDART(cudaMemcpy(data_dst, data_src, sizeof(VT) * rlen, cudaMemcpyDeviceToDevice));
                }
                DataObjectFactory::destroy(slice);
                break;
            }
            case VectorCombine::ADD: {
                if(localAddRes[o] == nullptr) {
                    // take lres and reset it to nullptr
                    localAddRes[o] = localResults[o];
                    localResults[o] = nullptr;
                }
                else {
                    CUDA::ewBinaryMat(BinaryOpCode::ADD, localAddRes[o], localAddRes[o], localResults[o], nullptr);
                }
                break;
            }
            default: {
                throw std::runtime_error(("VectorCombine case `"
                                          + std::to_string(static_cast<int64_t>(_data._combines[o])) + "` not supported"));
            }
        }
    }
}

template class CompiledPipelineTaskCUDA<DenseMatrix<double>>;
template class CompiledPipelineTaskCUDA<DenseMatrix<float>>;