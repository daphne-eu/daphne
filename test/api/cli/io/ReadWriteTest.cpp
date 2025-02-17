/*
 * Copyright 2024 The DAPHNE Consortium
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

#include <filesystem>
#include <parser/metadata/MetaDataParser.h>
#include <runtime/local/io/FileMetaData.h>
#include <string>

const std::string dirPath = "test/api/cli/io/";

// ********************************************************************************
// Read test cases
// ********************************************************************************

// These test cases check if the CSV reader reads the expected data from a CSV file.
// The data is chosen to trigger various interesting cases. The tests perform the following steps:
// - read a matrix/frame from a specified CSV file
// - compare it to a matrix/frame hard-coded in a DaphneDSL script (nan-safe for floating-point value types)
// - check if the number of incorrect elements is zero
#define MAKE_READ_TEST_CASE(dt, name, nanSafe)                                                                         \
    TEST_CASE("read_" dt "_" name, TAG_IO) {                                                                           \
        const std::string scriptPath = dirPath + "read/read_" + dt + "_" + name + ".daphne";                           \
        const std::string inPath = dirPath + "ref/" + dt + "_" + name + "_ref.csv";                                    \
        compareDaphneToStr("0\n", scriptPath.c_str(), "--args",                                                        \
                           ("inPath=\"" + inPath + "\",nanSafe=" + nanSafe).c_str());                                  \
    }

MAKE_READ_TEST_CASE("matrix", "si64", "false")
MAKE_READ_TEST_CASE("matrix", "f64", "true")
MAKE_READ_TEST_CASE("matrix", "str", "false")
MAKE_READ_TEST_CASE("frame", "mixed-no-str", "false")
MAKE_READ_TEST_CASE("frame", "mixed-str", "false")

// These test cases check if the CSV reader can be used in various relevant ways in DaphneDSL.
// The data is rather simple. The tests perform the following steps:
// - read a matrix/frame from a specified CSV file
// - compare it to a matrix/frame hard-coded in a DaphneDSL script
// - check if the number of incorrect elements is zero
#define MAKE_READ_TEST_CASE_2(scriptName)                                                                              \
    TEST_CASE("read_" scriptName ".daphne", TAG_IO) {                                                                  \
        const std::string scriptPath = dirPath + "read/read_" + scriptName + ".daphne";                                \
        compareDaphneToStr("0\n", scriptPath.c_str());                                                                 \
    }

// TODO The commented out test cases don't work yet (see #931).
MAKE_READ_TEST_CASE_2("matrix_read-in-udf")
MAKE_READ_TEST_CASE_2("matrix_dynamic-path-1")
// MAKE_READ_TEST_CASE_2("matrix_dynamic-path-2")
// MAKE_READ_TEST_CASE_2("matrix_dynamic-path-3")
MAKE_READ_TEST_CASE_2("frame_read-in-udf")
MAKE_READ_TEST_CASE_2("frame_dynamic-path-1")
// MAKE_READ_TEST_CASE_2("frame_dynamic-path-2")
// MAKE_READ_TEST_CASE_2("frame_dynamic-path-3")

TEST_CASE("readFrameFromCSVBinOpt", TAG_IO) {
    std::string filename = dirPath + "ref/ReadCsv1-1.csv";
    std::filesystem::remove(filename + ".dbdf");
    compareDaphneToRef(dirPath + "out/testReadFrameWithNoMeta.txt", dirPath + "read/testReadFrameWithNoMeta.daphne", "--second-read-opt");
    REQUIRE(std::filesystem::exists(filename + ".dbdf"));
    compareDaphneToRef(dirPath + "out/testReadFrameWithNoMeta.txt", dirPath + "read/testReadFrameWithNoMeta.daphne", "--second-read-opt");
    std::filesystem::remove(filename + ".dbdf");
}

TEST_CASE("readMatrixFromCSVBinOpt", TAG_IO) {
    std::string filename = dirPath + "ref/matrix_si64_ref.csv";
    std::filesystem::remove(filename + ".dbdf");
    compareDaphneToRef(dirPath + "out/testReadStringIntoFrameNoMeta.txt", dirPath + "read/testReadFrameWithMixedTypes.daphne", "--second-read-opt");
    REQUIRE(std::filesystem::exists(filename + ".dbdf"));
    compareDaphneToRef(dirPath + "out/testReadStringIntoFrameNoMeta.txt", dirPath + "read/testReadFrameWithMixedTypes.daphne", "--second-read-opt");
    std::filesystem::remove(filename + ".dbdf");
}

// ********************************************************************************
// Write test cases
// ********************************************************************************

// These test cases check if the CSV writer produces the expected CSV files.
// The data is chosen to trigger various interesting cases. The tests perform the following steps:
// - start with matrix/frame hard-coded in a DaphneDSL script
// - write it to a CSV file
// - don't compare the file contents to a reference file, as there can be multiple equivalent CSV representations,
//   because
//   - any field may be quoted, but only certain fields need to be quoted
//   - there could be trailing zeros in floating-point numbers
//   - there are different valid notations for floating-point numbers (e.g., 0.001 vs 1e-3)
//   - there are different valid notations for integers (e.g., 32 vs 0x20)
//   - ...
// - instead, read the written file and a reference CSV file and compare the two read matrices/frames in DaphneDSL
// - note: these tests don't check if the matrix/frame read from the reference CSV file looks as expected (that's done
//   by the read tests)
#define MAKE_WRITE_TEST_CASE(dt, name, nanSafe)                                                                        \
    TEST_CASE("write_" dt "_" name, TAG_IO) {                                                                          \
        const std::string scriptPathWrt = dirPath + "write/write_" + dt + "_" + name + ".daphne";                      \
        const std::string scriptPathCmp = dirPath + "do_check_" + dt + ".daphne";                                      \
        const std::string outPath = dirPath + "out/" + dt + "_" + name + ".csv";                                       \
        const std::string refPath = dirPath + "ref/" + dt + "_" + name + "_ref.csv";                                   \
        std::filesystem::remove(outPath); /* remove old output file if it still exists */                              \
        checkDaphneStatusCode(StatusCode::SUCCESS, scriptPathWrt.c_str(), "--args",                                    \
                              ("outPath=\"" + outPath + "\"").c_str());                                                \
        /* TODO REQUIRE() the status code to be success (don't only CHECK() it), because in case of failure, */        \
        /*      the next check doesn't make sense and might produce a misleading output. */                            \
        compareDaphneToStr("0\n", scriptPathCmp.c_str(), "--args",                                                     \
                           ("chkPath=\"" + outPath + "\",refPath=\"" + refPath + "\",nanSafe=" + nanSafe).c_str());    \
    }

// TODO The commented out test cases don't work yet, as the CSV writer doesn't support strings yet.
MAKE_WRITE_TEST_CASE("matrix", "si64", "false")
MAKE_WRITE_TEST_CASE("matrix", "f64", "true")
MAKE_WRITE_TEST_CASE("matrix", "str", "false")
MAKE_WRITE_TEST_CASE("matrix", "view", "false")
MAKE_WRITE_TEST_CASE("frame", "mixed-no-str", "false")
MAKE_WRITE_TEST_CASE("frame", "mixed-str", "false")