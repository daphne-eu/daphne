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

#ifndef SRC_RUNTIME_LOCAL_VECTORIZED_TASKQUEUES_H
#define SRC_RUNTIME_LOCAL_VECTORIZED_TASKQUEUES_H

#include <list>
#include <mutex>
#include <condition_variable>
#include <runtime/local/vectorized/Tasks.h>

const uint64_t DEFAULT_MAX_SIZE = 100000;

class TaskQueue {
public:
    virtual ~TaskQueue() = default;

    virtual void enqueueTask(Task* t) = 0;
    virtual void enqueueTaskOnTargetQueue(Task* t, int targetCPU) = 0;
    virtual Task* dequeueTask() = 0;
    virtual uint64_t size() = 0;
    virtual void closeInput() = 0;
};

class BlockingTaskQueue : public TaskQueue {
private:
    std::list<Task*> _data;
    std::mutex _qmutex;
    std::condition_variable _cv;
    EOFTask _eof; //end marker
    uint64_t _capacity;
    bool _closedInput;

public:
    BlockingTaskQueue() : BlockingTaskQueue(DEFAULT_MAX_SIZE) {}
    explicit BlockingTaskQueue(uint64_t capacity) {
        _closedInput = false;
        _capacity = capacity;
    }
    ~BlockingTaskQueue() override = default;

    void enqueueTask(Task* t) override {
        // lock mutex, released at end of scope
        std::unique_lock<std::mutex> ul(_qmutex);
        // blocking wait until tasks dequeued
        while( _data.size() + 1 > _capacity )
            _cv.wait(ul);
        // add task to end of list
        _data.push_back(t);
        // notify blocked dequeue operations
        _cv.notify_one();
    }

    void enqueueTaskOnTargetQueue(Task* t, int targetCPU) override {
        // Change CPU pinning before enqueue to utilize NUMA first-touch policy
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(targetCPU, &cpuset);
        sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
        std::unique_lock<std::mutex> ul(_qmutex);
        while( _data.size() + 1 > _capacity )
            _cv.wait(ul);
        _data.push_back(t);
        _cv.notify_one();
    }

    Task* dequeueTask() override {
        // lock mutex, released at end of scope
        std::unique_lock<std::mutex> ul(_qmutex);
        // blocking wait for new tasks
        while( _data.empty() ) {
            if( _closedInput )
                return &_eof;
            else
                _cv.wait(ul);
        }
        // obtain next task
        Task* t = _data.front();
        _data.pop_front();
        _cv.notify_one();
        return t;
    }

    uint64_t size() override {
        std::unique_lock<std::mutex> lu(_qmutex);
        return _data.size();
    }

    void closeInput() override {
        std::unique_lock<std::mutex> lu(_qmutex);
        _closedInput = true;
        _cv.notify_all();
    }
};

#endif //SRC_RUNTIME_LOCAL_VECTORIZED_TASKQUEUES_H
