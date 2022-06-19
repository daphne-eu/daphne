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
#include <numeric>

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
    std::vector<TaskQueue*> _q;
    std::vector<int> _physical_ids;
    std::vector<int> _unique_threads;
    std::array<bool, 256> eofWorkers;
    bool _verbose;
    uint32_t _fid;
    uint32_t _batchSize;
    int _threadID;
    int _numQueues;
    int _queueMode;
    int _stealLogic;
    bool _pinWorkers;
public:
    // this constructor is to be used in practice
    WorkerCPU(std::vector<TaskQueue*> deques, std::vector<int> physical_ids, std::vector<int> unique_threads, bool verbose, uint32_t fid = 0, uint32_t batchSize = 100, int threadID = 0, int numQueues = 0, int queueMode = 0, int stealLogic = 0, bool pinWorkers = 0) : Worker(), _q(deques), _physical_ids(physical_ids), _unique_threads(unique_threads),
            _verbose(verbose), _fid(fid), _batchSize(batchSize), _threadID(threadID), _numQueues(numQueues), _queueMode(queueMode), _stealLogic(stealLogic), _pinWorkers(pinWorkers) {
        // at last, start the thread
        t = std::make_unique<std::thread>(&WorkerCPU::run, this);
    }
    
    ~WorkerCPU() override = default;

    void run() override {
        if (_pinWorkers) {
            // pin worker to CPU core
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(_threadID, &cpuset);
            sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
        }

        int currentDomain = _physical_ids[_threadID];
        int targetQueue = _threadID;
	if( _queueMode == 0 ) {
            targetQueue = 0;
        } else if ( _queueMode == 1) {
            targetQueue = currentDomain;
        } else if ( _queueMode == 2) {
            targetQueue = _threadID;
	} else {
            std::cout << "Error finding queue." << std::endl;
        }
        int startingQueue = targetQueue;
        
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

        // All tasks from own queue have completed. Now stealing from other queues.

        if( _numQueues > 1 ) {
        if( _stealLogic == 0) {
            // Stealing in sequential order
            
            targetQueue = (targetQueue+1)%_numQueues;
            
            while ( targetQueue != startingQueue ) {
                t = _q[targetQueue]->dequeueTask();
                if( isEOF(t) ) {
                    targetQueue = (targetQueue+1)%_numQueues;
                } else {
                    t->execute(_fid, _batchSize);
                    delete t;
                }
            }
        } else if ( _stealLogic == 1) {
            // Stealing in sequential order from same domain first
            if ( _queueMode == 2 ) {
            targetQueue = (targetQueue+1)%_numQueues;

            while ( targetQueue != startingQueue ) {
                if ( _physical_ids[targetQueue] == currentDomain ){
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
            }
            
            // No more tasks on this domain, now switching to other domain
            
            targetQueue = (targetQueue+1)%_numQueues;
            
            while ( targetQueue != startingQueue ) {
                    t = _q[targetQueue]->dequeueTask();
                    if( isEOF(t) ) {
                        targetQueue = (targetQueue+1)%_numQueues;
                    } else {
                        t->execute(_fid, _batchSize);
                        delete t;
                    }
            }
        } else if( _stealLogic == 2) {
            // stealing from random workers until all workers EOF
            
            eofWorkers.fill(false);
            while( std::accumulate(eofWorkers.begin(), eofWorkers.end(), 0) < _numQueues ) {
                targetQueue = rand() % _numQueues;
                if( eofWorkers[targetQueue] == false ) {
                    t = _q[targetQueue]->dequeueTask();
                    //std::cout << "Execute task stolen from: " << targetQueue << std::endl;
                    if( isEOF(t) ) {
                        eofWorkers[targetQueue] = true;
                    } else {
                        t->execute(_fid, _batchSize);
                        delete t;
                    }
                }
            }
            
        } else if ( _stealLogic == 3) {
            // stealing from random workers from same socket first
            int queuesThisDomain = 0;
            eofWorkers.fill(false);
            
            for( int i=0; i<_numQueues; i++ ) {
                if( _physical_ids[i] == currentDomain ) {
                    queuesThisDomain++;
                }
            }
            if ( _queueMode == 2 ) {
            while( std::accumulate(eofWorkers.begin(), eofWorkers.end(), 0) < queuesThisDomain ) {
                targetQueue = rand() % _numQueues;
                if( _physical_ids[targetQueue] == currentDomain) {
                    if( eofWorkers[targetQueue] == false ) {
                        t = _q[targetQueue]->dequeueTask();
                        if( isEOF(t) ) {
                            eofWorkers[targetQueue] = true;
                        } else {
                            t->execute(_fid, _batchSize);
                            delete t;
                        }
                    }
                }
            }
            }
            
            // all workers on same domain are EOF, now also allowing stealing from other domain
            // This could also be done by keeping a list of EOF workers on the other domain
            
            while ( std::accumulate(eofWorkers.begin(), eofWorkers.end(), 0) < _numQueues ) {
                targetQueue = rand() % _numQueues;
                // no need to check if they are on the other domain, because otherwise they would be EOF anyway
                if( eofWorkers[targetQueue] == false ) {
                    t = _q[targetQueue]->dequeueTask();
                    if( isEOF(t) ) {
                        eofWorkers[targetQueue] = true;
                    } else {
                        t->execute(_fid, _batchSize);
                        delete t;
                    }
                }
            }
        }
	}
        
        // No more tasks available anywhere
        if( _verbose )
            std::cerr << "WorkerCPU: received EOF, finalized." << std::endl;
    }
};

////entry point for std:thread
//static void runWorker(Worker* worker) {
//    worker->run();
//}
