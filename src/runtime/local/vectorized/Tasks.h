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
class SingleOpTask : public Task {
private:
    uint64_t _rl;    // row lower index
    uint64_t _ru;    // row upper index
    uint64_t _bsize; // batch size (data binding)

public:
    SingleOpTask(); //TODO void** inputs, void* output, function pointer
    SingleOpTask(uint64_t rl, uint64_t ru, uint64_t bsize) {
        _rl = rl;
        _ru = ru;
        _bsize = bsize;
    }

    ~SingleOpTask() override = default;

    void execute() override {
        //TODO basic implementation
    }
};

//TODO tasks for compiled pipelines

#endif //SRC_RUNTIME_LOCAL_VECTORIZED_TASKS_H
