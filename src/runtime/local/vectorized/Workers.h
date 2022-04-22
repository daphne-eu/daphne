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
    TaskQueue* _q;
    bool _verbose;
    uint32_t _fid;
    uint32_t _batchSize;
public:
    // this constructor is to be used in practice
    WorkerCPU(TaskQueue* tq, bool verbose, uint32_t fid = 0, uint32_t batchSize = 100) : Worker(), _q(tq),
            _verbose(verbose), _fid(fid), _batchSize(batchSize) {
        // at last, start the thread
        t = std::make_unique<std::thread>(&WorkerCPU::run, this);
    }
    
    ~WorkerCPU() override = default;

    void run() override {
        Task* t = _q->dequeueTask();

        while( !isEOF(t) ) {
            //execute self-contained task
            if( _verbose )
                std::cerr << "WorkerCPU: executing task." << std::endl;
            t->execute(_fid, _batchSize);
            delete t;
            //get next tasks (blocking)
            t = _q->dequeueTask();
        }
        if( _verbose )
            std::cerr << "WorkerCPU: received EOF, finalized." << std::endl;
    }
};

class WorkerCPUPerCPU : public Worker {
    std::vector<TaskQueue*> _q;
    std::vector<int> _numaDomains;
    bool _verbose;
    uint32_t _fid;
    uint32_t _batchSize;
    int _threadID;
    int _numaID;
    int _numQueues;
    int _queueMode;
public:
    // this constructor is to be used in practice
    WorkerCPUPerCPU(std::vector<TaskQueue*> deques, std::vector<int> numaDomains, bool verbose, uint32_t fid = 0, uint32_t batchSize = 100, int threadID = 0, int numQueues = 0, int queueMode = 0) : Worker(), _q(deques), _numaDomains(numaDomains),
            _verbose(verbose), _fid(fid), _batchSize(batchSize), _threadID(threadID), _numQueues(numQueues), _queueMode(queueMode) {
        // at last, start the thread
        t = std::make_unique<std::thread>(&WorkerCPUPerCPU::run, this);
    }
    
    ~WorkerCPUPerCPU() override = default;

    void run() override {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(_threadID, &cpuset);
        sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
        int targetQueue = _threadID;
        int currentDomain = _numaDomains[_threadID];
        if( _queueMode == 1) {
            targetQueue = currentDomain;
        }
        
        Task* t = _q[targetQueue]->dequeueTask();

        while( !isEOF(t) ) {
            //execute self-contained task
            if( _verbose )
                std::cerr << "WorkerCPU: executing task." << std::endl;
            t->execute(_fid, _batchSize);
            delete t;
            //get next tasks (blocking)
            t = _q[targetQueue]->dequeueTask();
        }
        
        // No more tasks on own queue, now switching to other queues on the same socket
        
        targetQueue = (targetQueue+1)%_numQueues;

        while (targetQueue != _threadID) {
            if ( _numaDomains[targetQueue] == currentDomain ){
                t = _q[targetQueue]->dequeueTask();
                if( isEOF(t) ) {
                    targetQueue = (targetQueue+1)%_numQueues;
                } else {
                    t->execute(_fid, _batchSize);
                    delete t;
                }
            } else {
                targetQueue = (targetQueue+1)%_numQueues;
            }
        }
        
        // No more tasks on this socket, now switching to other socket
        
        targetQueue = (targetQueue+1)%_numQueues;
        
        while (targetQueue != _threadID) {
            if ( _numaDomains[targetQueue] != currentDomain ){
                t = _q[targetQueue]->dequeueTask();
                if( isEOF(t) ) {
                    targetQueue = (targetQueue+1)%_numQueues;
                } else {
                    t->execute(_fid, _batchSize);
                    delete t;
                }
            } else {
                targetQueue = (targetQueue+1)%_numQueues;
            }
        }
        
        // No more tasks available

        if( _verbose )
            std::cerr << "WorkerCPU: received EOF, finalized." << std::endl;
    }
};

class WorkerCPUPerGroup : public Worker {
    std::vector<TaskQueue*> _q;
    std::vector<int> _numaDomains;
    bool _verbose;
    uint32_t _fid;
    uint32_t _batchSize;
    int _threadID;
    int _numaID;
    int _numQueues;
    int _queueMode;
public:
    // this constructor is to be used in practice
    WorkerCPUPerGroup(std::vector<TaskQueue*> deques, std::vector<int> numaDomains, bool verbose, uint32_t fid = 0, uint32_t batchSize = 100, int threadID = 0, int numQueues = 0, int queueMode = 0) : Worker(), _q(deques), _numaDomains(numaDomains),
            _verbose(verbose), _fid(fid), _batchSize(batchSize), _threadID(threadID), _numQueues(numQueues), _queueMode(queueMode) {
        // at last, start the thread
        t = std::make_unique<std::thread>(&WorkerCPUPerGroup::run, this);
    }
    
    ~WorkerCPUPerGroup() override = default;

    void run() override {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(_threadID, &cpuset);
        sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
        int targetQueue = _threadID;
        int currentDomain = _numaDomains[_threadID];
        if( _queueMode == 1) {
            targetQueue = currentDomain;
        }
        
        Task* t = _q[targetQueue]->dequeueTask();

        while( !isEOF(t) ) {
            //execute self-contained task
            if( _verbose )
                std::cerr << "WorkerCPU: executing task." << std::endl;
            t->execute(_fid, _batchSize);
            delete t;
            //get next tasks (blocking)
            t = _q[targetQueue]->dequeueTask();
        }
        
        // No more tasks on own queue, now switching to other queues
        // Can be improved by assigning a "Foreman" for each socket
        // responsible for task stealing
        
        targetQueue = (targetQueue+1)%_numQueues;

        while(targetQueue != currentDomain) {
            t = _q[targetQueue]->dequeueTask();
            if( isEOF(t) ) {
                targetQueue = (targetQueue+1)%_numQueues;
            } else {
                t->execute(_fid, _batchSize);
                delete t;
            }
        }

        if( _verbose )
            std::cerr << "WorkerCPU: received EOF, finalized." << std::endl;
    }
};

////entry point for std:thread
//static void runWorker(Worker* worker) {
//    worker->run();
//}
