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

#ifndef SRC_RUNTIME_LOCAL_DATASTRUCTURES_DATAOBJECTFACTORY_H
#define SRC_RUNTIME_LOCAL_DATASTRUCTURES_DATAOBJECTFACTORY_H

#include <stdexcept>

struct DataObjectFactory {

    /**
     * @brief The central function for creating data objects of any data type
     * (matrices, frames).
     *
     * The arguments must match those of any (private) constructor of the
     * specified data type.
     *
     * @param args
     * @return
     */
    template <class DataType, typename... ArgTypes> static DataType *create(ArgTypes... args) {
        // TODO Employ placement-new.
        return new DataType(args...);
    }

    /**
     * @brief The central function for destroying data objects of any data type
     * (matrices, frames).
     *
     * Decreases the reference counter of the given data object. If the
     * reference counter becomes zero, the data object is destroyed.
     *
     * The access is protected by a mutex, such that multiple threads may call
     * this method concurrently.
     *
     * @param obj The data object to destroy.
     */
    template <class DataType> static void destroy(const DataType *obj) {
        if (!obj)
            throw std::runtime_error("DataObjectFactory::destroy() must not be called with nullptr");

        obj->refCounterMutex.lock();
        obj->refCounter--;
        if (obj->refCounter == 0) {
            obj->refCounterMutex.unlock();
            delete obj;
        } else
            obj->refCounterMutex.unlock();
    }

    // TODO Simplify many places in the code (especially test cases) by using
    // the new feature of destroying multiple data objects by one call.
    template <typename DataType, typename... Rest> static void destroy(const DataType *obj, const Rest *...rest) {
        destroy(obj);
        destroy(rest...);
    }
};

#endif // SRC_RUNTIME_LOCAL_DATASTRUCTURES_DATAOBJECTFACTORY_H