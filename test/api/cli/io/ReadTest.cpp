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

#include <api/cli/Utils.h>

#include <tags.h>

#include <catch.hpp>

#include <sstream>
#include <string>

const std::string dirPath = "test/api/cli/io/";

// TODO Add script-level test cases for reading files in various formats.
// This test case used to read a COO file, but being a quick fix, it was not
// integrated cleanly. There should either be a seprate reader for COO, or we
// should do that via the respective Matrix Market format.
#if 0
TEST_CASE("readSparse", TAG_IO) {
    auto arg = "filename=\"" + dirPath + "readSparse.coo\"";
    compareDaphneToRef(dirPath + "readSparse.txt",
        dirPath + "readSparse.daphne",
        "--select-matrix-representations",
        "--args",
        arg.c_str());
}
#endif

TEST_CASE("readFrameFromCSV", TAG_IO)
{
    compareDaphneToRef(dirPath + "testReadFrame.txt", dirPath + "testReadFrame.daphne");
}

TEST_CASE("readMatrixFromCSV", TAG_IO)
{
    compareDaphneToRef(dirPath + "testReadMatrix.txt", dirPath + "testReadMatrix.daphne");
}

// does not yet work!
// TEST_CASE("readReadMatrixFromCSV_DynamicPath", TAG_IO)
// {
//     compareDaphneToRef(dirPath + "testReadMatrix.txt", dirPath + "testReadMatrix_DynamicPath.daphne");
// }