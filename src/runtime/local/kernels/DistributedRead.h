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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_DISTRIBUTEDREAD_H
#define SRC_RUNTIME_LOCAL_KERNELS_DISTRIBUTEDREAD_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/Handle.h>
#include <runtime/local/io/ReadCsv.h>
#include <runtime/local/io/File.h>
#include <cassert>
#include <cstddef>



// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DT>
void distributedRead(Handle<DT> *&res, const char * filename, DCTX(ctx))
{
    FileMetaData fmd = FileMetaData::ofFile(filename);    

    readCsv(res, filename, fmd.numRows, fmd.numCols, ',');
}


#endif //SRC_RUNTIME_LOCAL_KERNELS_DISTRIBUTEDREAD_H
