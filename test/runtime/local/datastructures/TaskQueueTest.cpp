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

#include <runtime/local/vectorized/TaskQueues.h>
#include <tags.h>
#include <catch.hpp>
#include <cstdint>

TEST_CASE("Task sequence", TAG_DATASTRUCTURES) {
    TaskQueue* bq = new BlockingTaskQueue(5);
    Task* t1 = new SingleOpTask<double>( 0, 10, 1);
    Task* t2 = new SingleOpTask<double>(10, 20, 1);
    Task* t3 = new SingleOpTask<double>(20, 30, 1);
    
    //check return sequence
    bq->enqueueTask(t1);
    bq->enqueueTask(t2);
    bq->enqueueTask(t3);
    CHECK(bq->dequeueTask() == t1);
    CHECK(bq->dequeueTask() == t2);
    CHECK(bq->dequeueTask() == t3);

    delete t1;
    delete t2;
    delete t3;
    delete bq;
}

TEST_CASE("Queue size", TAG_DATASTRUCTURES) {
    TaskQueue* bq = new BlockingTaskQueue(5);
    Task* t1 = new SingleOpTask<double>( 0, 10, 1);
    Task* t2 = new SingleOpTask<double>(10, 20, 1);

    // check proper size management
    CHECK(bq->size() == 0);
    bq->enqueueTask(t1);
    CHECK(bq->size() == 1);
    bq->enqueueTask(t2);
    CHECK(bq->size() == 2);
    bq->dequeueTask();
    CHECK(bq->size() == 1);
    bq->dequeueTask();
    CHECK(bq->size() == 0);

    delete t1;
    delete t2;
    delete bq;
}

TEST_CASE("EOF handling", TAG_DATASTRUCTURES) {
    TaskQueue* bq = new BlockingTaskQueue(5);
    Task* t1 = new SingleOpTask<double>( 0, 10, 1);

    // check EOF after last task
    bq->enqueueTask(t1);
    bq->closeInput();
    bq->dequeueTask();
    CHECK(dynamic_cast<EOFTask*>(bq->dequeueTask()));
    CHECK(dynamic_cast<EOFTask*>(bq->dequeueTask()));

    delete t1;
    delete bq;
}
