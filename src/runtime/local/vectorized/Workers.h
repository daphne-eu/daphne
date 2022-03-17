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

#include <thread>
#include <sched.h>
#include <fstream>

class Worker {
protected:
    std::unique_ptr<std::thread> t;

    // Worker only used as derived class, which starts the thread after the class has been constructed (order matters).
    Worker() : t() {}
public:
    // Worker is move only due to std::thread. Therefore, we delete the copy constructor and the assignment operator.
    Worker(const Worker&) = delete;
    Worker& operator=(const Worker&) = delete;

    // move constructor
    Worker(Worker&& obj)  noexcept : t(std::move(obj.t)) {}

    // move assignment operator
    Worker& operator=(Worker&& obj)  noexcept {
        if(t->joinable())
            t->join();
        t = std::move(obj.t);
        return *this;
    }

    virtual ~Worker() {
        if(t->joinable())
            t->join();
    };

    void join() {
        t->join();
    }
    virtual void run() = 0;
    static bool isEOF(Task* t) {
        return dynamic_cast<EOFTask*>(t);
    }
};

class WorkerCPU : public Worker {
    //TaskQueue* _q[64];
    std::vector<TaskQueue*> _q;
    bool _verbose;
    uint32_t _fid;
    uint32_t _batchSize;
    int _threadID;
    int _numaID;
    int _numQueues;
    int _queueMode;
public:
    // this constructor is to be used in practice
    WorkerCPU(std::vector<TaskQueue*> deques, bool verbose, uint32_t fid = 0, uint32_t batchSize = 100, int threadID = 0, int numQueues = 0, int queueMode = 0) : Worker(), _q(deques),
            _verbose(verbose), _fid(fid), _batchSize(batchSize), _threadID(threadID), _numQueues(numQueues), _queueMode(queueMode) {
        // at last, start the thread
        t = std::make_unique<std::thread>(&WorkerCPU::run, this);
    }
    
    ~WorkerCPU() override = default;

    void run() override {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(_threadID, &cpuset);
        sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
        int target_deque = 0;
        std::vector<Task*> tmp;

        if( _queueMode == 0 ) {
            target_deque = 0;
        } else if( _queueMode == 1) {
            std::string line;
            std::ifstream cpuNodeFile("/sys/devices/system/cpu/cpu" + std::to_string(_threadID) + "/topology/physical_package_id");
            std::getline(cpuNodeFile, line);
            cpuNodeFile.close();
            _numaID = stoi(line);
            target_deque = _numaID;
        } else if( _queueMode == 2) {
            target_deque = _threadID;
        }

        Task* t = _q[target_deque]->dequeueTask();

        while( !isEOF(t) ) {
            if( _verbose )
                std::cerr << "WorkerCPU: executing task." << std::endl;
            t->execute(_fid, _batchSize);
            delete t;

            //get next tasks (blocking)
            t = _q[target_deque]->dequeueTask();
        }

        // Reached the end of own queue, now attempting to steal from other queues
        if( _numQueues > 1 ) {
        target_deque = (target_deque+1)%_numQueues;
        if( _queueMode == 1) {
            while(target_deque != _numaID) {
                t = _q[target_deque]->dequeueTask();
                if( isEOF(t) ) {
                    target_deque = (target_deque+1)%_numQueues;
                } else {
                    t->execute(_fid, _batchSize);
                    delete t;
                    }
                }
            }
        } else if( _queueMode == 2 ) {
            while(target_deque != _threadID) {
                if(_q[target_deque]->dequeueBatch(tmp, _numQueues) > 0) {
                    _q[_threadID]->enqueueBatch(tmp);
                }
                tmp.clear();
                t = _q[_threadID]->dequeueTask();
                while( !isEOF(t) ) {
                    t->execute(_fid, _batchSize);
                    delete t;
                    t = _q[_threadID]->dequeueTask();
                }
                target_deque = (target_deque+1)%_numQueues;
            }
            delete t;
        }

        if( _verbose )
            std::cerr << "WorkerCPU: received EOF on all queues, finalized." << std::endl;
    }
};

////entry point for std:thread
//static void runWorker(Worker* worker) {
//    worker->run();
//}
