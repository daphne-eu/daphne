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

#ifndef SRC_RUNTIME_LOCAL_VECTORIZED_TASKS_H
#define SRC_RUNTIME_LOCAL_VECTORIZED_TASKS_H

#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/EwBinaryMat.h>
#include <runtime/local/vectorized/VectorizedDataSink.h>
#include <runtime/local/context/DaphneContext.h>
#include <ir/daphneir/Daphne.h>

#include <functional>
#include <vector>
#include <mutex>

using mlir::daphne::VectorSplit;
using mlir::daphne::VectorCombine;

class Task {
public:
    virtual ~Task() = default;

    virtual void execute() = 0;
};

// task for signaling closed input queue (no more tasks)
class EOFTask : public Task {
public:
    EOFTask() = default;
    ~EOFTask() override = default;
    void execute() override {}
};

// Deprecated
// single operation task (multi-threaded operations)
template <class VT>
class SingleOpTask : public Task {
private:
    void (*_func)(DenseMatrix<VT>*,DenseMatrix<VT>*,DenseMatrix<VT>*);
    DenseMatrix<VT>* _res;
    DenseMatrix<VT>* _input1;
    DenseMatrix<VT>* _input2;
    uint64_t _rl{};    // row lower index
    uint64_t _ru{};    // row upper index
    uint64_t _bsize{}; // batch size (data binding)

public:
    SingleOpTask() = default;

    SingleOpTask(uint64_t rl, uint64_t ru, uint64_t bsize) :
        SingleOpTask(nullptr, nullptr, nullptr, nullptr, rl, ru, bsize) {}

    SingleOpTask(void (*func)(DenseMatrix<VT>*,DenseMatrix<VT>*,DenseMatrix<VT>*),
        DenseMatrix<VT>* res, DenseMatrix<VT>* input1, DenseMatrix<VT>* input2,
        uint64_t rl, uint64_t ru, uint64_t bsize)
    {
        _func = func;
        _res = res;
        _input1 = input1;
        _input2 = input2;
        _rl = rl;
        _ru = ru;
        _bsize = bsize;
    }

    ~SingleOpTask() override = default;

    void execute() override {
       for( uint64_t r = _rl; r < _ru; r+=_bsize ) {
           //create zero-copy views of inputs/outputs
           uint64_t r2 = std::min(r+_bsize, _ru);
           DenseMatrix<VT>* lres = _res->slice(r, r2);
           DenseMatrix<VT>* linput1 = _input1->slice(r, r2);
           DenseMatrix<VT>* linput2 = (_input2->getNumRows()==1) ?
               _input2 : _input2->slice(r, r2); //broadcasting
           //execute function on given data binding (batch size)
           _func(lres, linput1, linput2);
           //cleanup
           //TODO can't delete views without destroying the underlying arrays + private
       }
    }
};

template<class DT>
struct CompiledPipelineTaskData {
    std::function<void(DT ***, Structure **, DCTX(ctx))> _func;
    Structure **_inputs;
    size_t _numInputs;
    size_t _numOutputs;
    int64_t *_outRows;
    int64_t *_outCols;
    VectorSplit *_splits;
    VectorCombine *_combines;
    uint64_t _rl;    // row lower index
    uint64_t _ru;    // row upper index
    uint64_t _bsize; // batch size (data binding)
    int64_t *_wholeResultRows; // number of rows of the complete result
    int64_t *_wholeResultCols; // number of cols of the complete result
    uint64_t _offset;
    DCTX(_ctx);

    CompiledPipelineTaskData<DT> withDifferentRange(uint64_t newRl, uint64_t newRu) {
        CompiledPipelineTaskData<DT> flatCopy = *this;
        flatCopy._rl = newRl;
        flatCopy._ru = newRu;
        return flatCopy;
    }
};

template<class DT>
class CompiledPipelineTaskBase : public Task {
protected:
    CompiledPipelineTaskData<DT> _data;

public:
    explicit CompiledPipelineTaskBase(CompiledPipelineTaskData<DT> data) : _data(data) {}
    void execute() override = 0;

protected:
    bool isBroadcast(mlir::daphne::VectorSplit splitMethod, Structure *input) {
        return splitMethod == VectorSplit::NONE || (splitMethod == VectorSplit::ROWS && input->getNumRows() == 1);
    }

    std::vector<Structure *> createFuncInputs(uint64_t rowStart, uint64_t rowEnd) {
        std::vector<Structure *> linputs;
        for(auto i = 0u ; i < _data._numInputs ; i++) {
            if (isBroadcast(_data._splits[i], _data._inputs[i])) {
                linputs.push_back(_data._inputs[i]);
            }
            else if (VectorSplit::ROWS == _data._splits[i]) {
                linputs.push_back(_data._inputs[i]->slice(rowStart, rowEnd));
            }
            else {
                llvm_unreachable("Not all vector splits handled");
            }
        }
        return linputs;
    }

    void cleanupFuncInputs(std::vector<Structure *> &&linputs) {
        for(auto i = 0u ; i < _data._numInputs ; i++) {
            if(_data._inputs[i] != linputs[i]) {
                // slice copy was created
                DataObjectFactory::destroy(linputs[i]);
            }
        }
    }
};

template<class DT>
class CompiledPipelineTask : public CompiledPipelineTaskBase<DT> {};

template<typename VT>
class CompiledPipelineTask<DenseMatrix<VT>> : public CompiledPipelineTaskBase<DenseMatrix<VT>> {
    std::mutex &_resLock;
    DenseMatrix<VT> ***_res;
    using CompiledPipelineTaskBase<DenseMatrix<VT>>::_data;
public:
    CompiledPipelineTask(CompiledPipelineTaskData<DenseMatrix<VT>> data, std::mutex &resLock, DenseMatrix<VT> ***res)
        : CompiledPipelineTaskBase<DenseMatrix<VT>>(data), _resLock(resLock), _res(res) {}

    void execute() override;

private:
    void accumulateOutputs(std::vector<DenseMatrix<VT> *> &localResults,
                           std::vector<DenseMatrix<VT> *> &localAddRes,
                           uint64_t rowStart,
                           uint64_t rowEnd);
};

template<typename VT>
class CompiledPipelineTask<CSRMatrix<VT>> : public CompiledPipelineTaskBase<CSRMatrix<VT>> {
    // TODO: multiple sinks
    VectorizedDataSink<CSRMatrix<VT>> &_resultSink;
    using CompiledPipelineTaskBase<CSRMatrix<VT>>::_data;
public:
    CompiledPipelineTask(CompiledPipelineTaskData<CSRMatrix<VT>> data, VectorizedDataSink<CSRMatrix<VT>> &resultSink)
        : CompiledPipelineTaskBase<CSRMatrix<VT>>(data), _resultSink(resultSink) {}

    void execute() override {
        assert(_data._numOutputs == 1 && "TODO");
        size_t localResNumRows;
        size_t localResNumCols;
        switch(_data._combines[0]) {
        case VectorCombine::ROWS: {
            assert(_data._wholeResultCols[0] != -1 && "TODO");
            localResNumRows = _data._ru - _data._rl;
            localResNumCols = _data._wholeResultCols[0];
            break;
        }
        case VectorCombine::COLS: {
            assert(_data._wholeResultRows[0] != -1 && "TODO");
            localResNumRows = _data._wholeResultRows[0];
            localResNumCols = _data._ru - _data._rl;
            break;
        }
        default:
            throw std::runtime_error("Not implemented");
        }

        VectorizedDataSink<CSRMatrix<VT>> localSink(_data._combines[0], localResNumRows, localResNumCols);
        CSRMatrix<VT> *lres = nullptr;
        for(uint64_t r = _data._rl ; r < _data._ru ; r += _data._bsize) {
            //create zero-copy views of inputs/outputs
            uint64_t r2 = std::min(r + _data._bsize, _data._ru);

            auto linputs = this->createFuncInputs(r, r2);
            CSRMatrix<VT> **outputs[] = {&lres};
            //execute function on given data binding (batch size)
            _data._func(outputs, linputs.data(), _data._ctx);
            localSink.add(lres, r - _data._rl, false);

            // cleanup
            lres = nullptr;
            this->cleanupFuncInputs(std::move(linputs));
        }
        _resultSink.add(localSink.consume(), _data._rl);
    }
};

#endif //SRC_RUNTIME_LOCAL_VECTORIZED_TASKS_H
