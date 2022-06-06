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

#include "runtime/local/vectorized/Tasks.h"
#include "runtime/local/kernels/EwBinaryMat.h"

template<typename VT>
void CompiledPipelineTask<DenseMatrix<VT>>::execute(uint32_t fid, uint32_t batchSize) {
    // local add aggregation to minimize locking
    std::vector<DenseMatrix<VT>*> localAddRes(_data._numOutputs);
    std::vector<DenseMatrix<VT>*> localResults(_data._numOutputs);
    std::vector<DenseMatrix<VT>**> outputs;
    for (auto &lres : localResults)
        outputs.push_back(&lres);
    for(uint64_t r = _data._rl ; r < _data._ru ; r += batchSize) {
        //create zero-copy views of inputs/outputs
        uint64_t r2 = std::min(r + batchSize, _data._ru);
        
        auto linputs = this->createFuncInputs(r, r2);
        
        //execute function on given data binding (batch size)
        _data._funcs[fid](outputs.data(), linputs.data(), _data._ctx);
        accumulateOutputs(localResults, localAddRes, r, r2);
        
        // cleanup
        for (auto &localResult : localResults)
            if(localResult) {
                DataObjectFactory::destroy(localResult);
                localResult = nullptr;
            }
        
        // Note that a pipeline manages the reference counters of its inputs
        // internally. Thus, we do not need to care about freeing the inputs
        // here.
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
                ewBinaryMat(BinaryOpCode::ADD, result, result, localAddRes[o], _data._ctx);
                _resLock.unlock();
                //cleanup
                DataObjectFactory::destroy(localAddRes[o]);
            }
        }
    }
}

template<typename VT>
void CompiledPipelineTask<DenseMatrix<VT>>::accumulateOutputs(std::vector<DenseMatrix<VT> *> &localResults,
        std::vector<DenseMatrix<VT> *> &localAddRes, uint64_t rowStart, uint64_t rowEnd) {
    //TODO: in-place computation via better compiled pipelines
    //TODO: multi-return
    for(auto o = 0u ; o < _data._numOutputs ; ++o) {
        auto &result = (*_res[o]);
        switch (_data._combines[o]) {
            case VectorCombine::ROWS: {
                rowStart -= _data._offset;
                rowEnd -= _data._offset;
                auto slice = result->sliceRow(rowStart, rowEnd);
                // TODO It's probably more efficient to memcpy than to get/set.
                // But eventually, we don't want to copy at all.
                for(auto i = 0u ; i < slice->getNumRows() ; ++i) {
                    for(auto j = 0u ; j < slice->getNumCols() ; ++j) {
                        slice->set(i, j, localResults[o]->get(i, j));
                    }
                }
                DataObjectFactory::destroy(slice);
                break;
            }
            case VectorCombine::COLS: {
                auto slice = result->sliceCol(rowStart-_data._offset, rowEnd-_data._offset);
                // TODO It's probably more efficient to memcpy than to get/set.
                // But eventually, we don't want to copy at all.
                for(auto i = 0u ; i < slice->getNumRows() ; ++i) {
                    for(auto j = 0u ; j < slice->getNumCols() ; ++j) {
                        slice->set(i, j, localResults[o]->get(i, j));
                    }
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
                    ewBinaryMat(BinaryOpCode::ADD, localAddRes[o], localAddRes[o], localResults[o], nullptr);
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

template<typename VT>
void CompiledPipelineTask<CSRMatrix<VT>>::execute(uint32_t fid, uint32_t batchSize) {
    std::vector<size_t> localResNumRows(_data._numOutputs);
    std::vector<size_t> localResNumCols(_data._numOutputs);
    for(size_t i = 0; i < _data._numOutputs; i++) {
        switch(_data._combines[i]) {
            case VectorCombine::ROWS: {
                assert(_data._wholeResultCols[i] != -1 && "TODO");
                localResNumRows[i] = _data._ru - _data._rl;
                localResNumCols[i] = _data._wholeResultCols[i];
                break;
            }
            case VectorCombine::COLS: {
                assert(_data._wholeResultRows[i] != -1 && "TODO");
                localResNumRows[i] = _data._wholeResultRows[i];
                localResNumCols[i] = _data._ru - _data._rl;
                break;
            }
            default:
                throw std::runtime_error("Not implemented");
        }
    }
    
    std::vector<VectorizedDataSink<CSRMatrix<VT>>*> localSinks(_data._numOutputs);
    for(size_t i = 0; i < _data._numOutputs; i++)
        localSinks[i] = new VectorizedDataSink<CSRMatrix<VT>>(_data._combines[i], localResNumRows[i], localResNumCols[i]);
    
    std::vector<CSRMatrix<VT>*> lres(_data._numOutputs, nullptr);
    for(uint64_t r = _data._rl ; r < _data._ru ; r += batchSize) {
        //create zero-copy views of inputs/outputs
        uint64_t r2 = std::min(r + batchSize, _data._ru);
        
        auto linputs = this->createFuncInputs(r, r2);
        CSRMatrix<VT> *** outputs = new CSRMatrix<VT>**[_data._numOutputs];
        for(size_t i = 0; i < _data._numOutputs; i++)
            outputs[i] = &(lres[i]);
        //execute function on given data binding (batch size)
        _data._funcs[fid](outputs, linputs.data(), _data._ctx);
        delete[] outputs;
        for(size_t i = 0; i < _data._numOutputs; i++)
            localSinks[i]->add(lres[i], r - _data._rl, false);

        // cleanup
        for(size_t i = 0; i < _data._numOutputs; i++)
            lres[i] = nullptr;
        
        // Note that a pipeline manages the reference counters of its inputs
        // internally. Thus, we do not need to care about freeing the inputs
        // here.
    }
    for(size_t i = 0; i < _data._numOutputs; i++) {
        _resultSinks[i]->add(localSinks[i]->consume(), _data._rl);
        delete localSinks[i];
    }
}

template class CompiledPipelineTask<DenseMatrix<double>>;
template class CompiledPipelineTask<DenseMatrix<float>>;

template class CompiledPipelineTask<CSRMatrix<double>>;
template class CompiledPipelineTask<CSRMatrix<float>>;
