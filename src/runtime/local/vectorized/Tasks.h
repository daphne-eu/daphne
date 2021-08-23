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
#include <functional>

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
           uint64_t r2 = std::max(r+_bsize, _ru);
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
template <class VT>
class CompiledPipelineTask : public Task {
private:
    std::function<void(DenseMatrix<VT>***, DenseMatrix<VT>**)> _func;
    DenseMatrix<VT>* _res;
    DenseMatrix<VT>* _input1;
    DenseMatrix<VT>* _input2;
    uint64_t _rl;    // row lower index
    uint64_t _ru;    // row upper index
    uint64_t _bsize; // batch size (data binding)

public:
    CompiledPipelineTask();

    CompiledPipelineTask(uint64_t rl, uint64_t ru, uint64_t bsize) :
        CompiledPipelineTask(nullptr, nullptr, nullptr, nullptr, rl, ru, bsize) {}

    CompiledPipelineTask(std::function<void(DenseMatrix<VT>***, DenseMatrix<VT>**)> func,
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

    ~CompiledPipelineTask() override = default;

    void execute() override {
        for( uint64_t r = _rl; r < _ru; r+=_bsize ) {
            //create zero-copy views of inputs/outputs
            uint64_t r2 = std::max(r+_bsize, _ru);
            DenseMatrix<VT>* lres = nullptr;
            DenseMatrix<VT>* linput1 = _input1->slice(r, r2);
            DenseMatrix<VT>* linput2 = (_input2->getNumRows()==1) ?
                                       _input2 : _input2->slice(r, r2); //broadcasting

            DenseMatrix<VT> **outputs[] = {&lres};
            DenseMatrix<VT>* inputs[] = {linput1, linput2};
            //execute function on given data binding (batch size)
            _func(outputs, inputs);
            //TODO: in-place computation via better compiled pipelines
            auto slice = _res->slice(_rl, r2);
            for(auto i = 0u; i < slice->getNumRows(); ++i) {
                for(auto j = 0u; j < slice->getNumCols(); ++j) {
                    slice->set(i, j, lres->get(i, j));
                }
            }
            //cleanup
            //TODO cant't delete views without destroying the underlying arrays + private
        }
    }
};

#endif //SRC_RUNTIME_LOCAL_VECTORIZED_TASKS_H
