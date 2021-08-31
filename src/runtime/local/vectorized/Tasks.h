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
#include <ir/daphneir/Daphne.h>

#include <functional>
#include <vector>

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
    EOFTask() {}
    ~EOFTask() {}
    void execute() override {}
};

// single operation task (multi-threaded operations)
template <class VT>
class SingleOpTask : public Task {
private:
	void (*_func)(DenseMatrix<VT>*,DenseMatrix<VT>*,DenseMatrix<VT>*);
    DenseMatrix<VT>* _res;
    DenseMatrix<VT>* _input1;
    DenseMatrix<VT>* _input2;
    uint64_t _rl;    // row lower index
    uint64_t _ru;    // row upper index
    uint64_t _bsize; // batch size (data binding)

public:
    SingleOpTask();

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
           //TODO cant't delete views without destroying the underlying arrays + private
       }
    }
};

//TODO tasks for compiled pipelines
template<class VT>
class CompiledPipelineTask : public Task
{
private:
    std::function<void(DenseMatrix<VT> ***, DenseMatrix<VT> **)> _func;
    std::mutex &_resLock;
    DenseMatrix<VT> *&_res;
    DenseMatrix<VT> **_inputs;
    size_t _numInputs;
    size_t _numOutputs;
    int64_t *_outRows;
    int64_t *_outCols;
    VectorSplit *_splits;
    VectorCombine *_combines;
    uint64_t _rl;    // row lower index
    uint64_t _ru;    // row upper index
    uint64_t _bsize; // batch size (data binding)

public:
    CompiledPipelineTask(std::function<void(DenseMatrix<VT> ***, DenseMatrix<VT> **)> func,
                         std::mutex &resLock,
                         DenseMatrix<VT> *&res,
                         DenseMatrix<VT> **inputs,
                         size_t numInputs,
                         size_t numOutputs,
                         int64_t *outRows,
                         int64_t *outCols,
                         VectorSplit *splits,
                         VectorCombine *combines,
                         uint64_t rl,
                         uint64_t ru,
                         uint64_t bsize)
        : _func(func), _resLock(resLock), _res(res), _inputs(inputs), _numInputs(numInputs), _numOutputs(numOutputs),
          _outRows(outRows), _outCols(outCols), _splits(splits), _combines(combines), _rl(rl), _ru(ru), _bsize(bsize)
    {}

    ~CompiledPipelineTask() override = default;

    void execute() override
    {
        // local add aggregation to minimize locking
        DenseMatrix<VT> *localAddRes = nullptr;
        DenseMatrix<VT> *lres = nullptr;
        for(uint64_t r = _rl; r < _ru; r += _bsize) {
            //create zero-copy views of inputs/outputs
            uint64_t r2 = std::min(r + _bsize, _ru);

            auto linputs = createFuncInputs(r, r2);
            DenseMatrix<VT> **outputs[] = {&lres};
            //execute function on given data binding (batch size)
            _func(outputs, linputs.data());
            accumulateOutputs(lres, localAddRes, r, r2);

            // cleanup
            DataObjectFactory::destroy(lres);
            lres = nullptr;
            for(auto i = 0u; i < _numInputs; i++) {
                if (_splits[i] == VectorSplit::ROWS && _inputs[i]->getNumRows() != 1) {
                    // slice copy was created
                    DataObjectFactory::destroy(linputs[i]);
                }
            }
        }

        if (_combines[0] == VectorCombine::ADD) {
            _resLock.lock();
            if (_res == nullptr) {
                _res = localAddRes;
                _resLock.unlock();
            }
            else {
                ewBinaryMat(BinaryOpCode::ADD, _res, _res, localAddRes, nullptr);
                _resLock.unlock();
                //cleanup
                DataObjectFactory::destroy(localAddRes);
            }
        }
    }

    void accumulateOutputs(DenseMatrix<VT> *&lres, DenseMatrix<VT> *&localAddRes, uint64_t rowStart, uint64_t rowEnd)
    {
        //TODO: in-place computation via better compiled pipelines
        //TODO: multi-return
        for(auto o = 0u; o < 1; ++o) {
            switch (_combines[o]) {
            case VectorCombine::ROWS: {
                auto slice = _res->slice(rowStart, rowEnd);
                for(auto i = 0u; i < slice->getNumRows(); ++i) {
                    for(auto j = 0u; j < slice->getNumCols(); ++j) {
                        slice->set(i, j, lres->get(i, j));
                    }
                }
                DataObjectFactory::destroy(slice);
                break;
            }
            case VectorCombine::COLS: {
                auto slice = _res->slice(0, _outRows[o], rowStart, rowEnd);
                for(auto i = 0u; i < slice->getNumRows(); ++i) {
                    for(auto j = 0u; j < slice->getNumCols(); ++j) {
                        slice->set(i, j, lres->get(i, j));
                    }
                }
                DataObjectFactory::destroy(slice);
                break;
            }
            case VectorCombine::ADD: {
                if (localAddRes == nullptr) {
                    // take lres and reset it to nullptr
                    localAddRes = lres;
                    lres = nullptr;
                }
                else {
                    ewBinaryMat(BinaryOpCode::ADD, localAddRes, localAddRes, lres, nullptr);
                }
                break;
            }
            default: {
                throw std::runtime_error(("VectorCombine case `"
                    + std::to_string(static_cast<int64_t>(_combines[o])) + "` not supported"));
            }
            }
        }
    }

    std::vector<DenseMatrix<VT> *> createFuncInputs(uint64_t rowStart, uint64_t rowEnd)
    {
        std::vector<DenseMatrix<VT> *> linputs;
        for(auto i = 0u; i < _numInputs; i++) {
            if (_splits[i] == VectorSplit::ROWS) {
                // broadcasting
                linputs.push_back((_inputs[i]->getNumRows() == 1) ? _inputs[i] : _inputs[i]->slice(rowStart, rowEnd));
            }
            else {
                linputs.push_back(_inputs[i]);
            }
        }
        return linputs;
    }
};

#endif //SRC_RUNTIME_LOCAL_VECTORIZED_TASKS_H
