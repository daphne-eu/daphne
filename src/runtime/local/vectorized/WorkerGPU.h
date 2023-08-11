/*
 * Copyright 2022 The DAPHNE Consortium
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

#include "Worker.h"
#include <runtime/local/vectorized/TaskQueues.h>

class WorkerGPU : public Worker {
    TaskQueue* _q;
    bool _verbose;
    uint32_t _fid;
    uint32_t _batchSize;
public:
    // ToDo: remove compile-time verbose parameter and use logger
    WorkerGPU(TaskQueue* tq, DCTX(dctx), bool verbose, uint32_t fid = 0, uint32_t batchSize = 100) : Worker(dctx), _q(tq),
            _verbose(verbose), _fid(fid), _batchSize(batchSize) {
        // at last, start the thread
        t = std::make_unique<std::thread>(&WorkerGPU::run, this);
    }

    ~WorkerGPU() override = default;

    void run() override {
        Task* t = _q->dequeueTask();

        while( !isEOF(t) ) {
            //execute self-contained task
            if( _verbose )
                ctx->logger->trace("WorkerGPU: executing task.");
            t->execute(_fid, _batchSize);
            delete t;
            //get next tasks (blocking)
            t = _q->dequeueTask();
        }
        if( _verbose )
            ctx->logger->trace("WorkerGPU: received EOF, finalized.");
    }
};