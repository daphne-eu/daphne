/*
 * Copyright 2025 The DAPHNE Consortium
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

#include <api/cli/StatusCode.h>
#include <api/cli/Utils.h>
#include <ir/daphneir/Daphne.h>

#include <tags.h>

#include <catch.hpp>

#include <sstream>
#include <string>
#include <vector>

const std::string dirPath = "test/api/cli/sql/";

/**
 * @brief Checks if the specified `requiredOps` are contained and if the specified `disallowedOps` are not contained in
 * the IR after lowering to columnar ops.
 *
 * Runs the specified script in DAPHNE (using --columnar if specified), checks for successful completion and correct
 * result, and checks for the specified required and disallowed ops in the IR. Besides the explicitly specified
 * disallowed ops, ops for the conversion between position lists and bit vectors are always implicitly disallowed,
 * because they shall not remain in the IR.
 *
 * @param exp The expected output on `cout`.
 * @param scriptFilePath The path to the DaphneDSL script to execute.
 * @param useColumnar Whether to invoke DAPHNE with the `--columnar` flag.
 * @param requiredOps List of DaphneIR op mnemonics that must be contained in the IR.
 * @param disallowedOps List of DaphneIR op mnemonics that must not be contained in the IR.
 */
void checkOps(const std::string &exp, const std::string &scriptFilePath, bool useColumnar,
              const std::vector<std::string> &requiredOps, const std::vector<std::string> &disallowedOps) {
    // Ops converting between position lists and bit vectors. These should never remain after simplificatin rewrites.
    std::vector<std::string> cnvOps = {mlir::daphne::ConvertPosListToBitmapOp::getOperationName().str(),
                                       mlir::daphne::ConvertBitmapToPosListOp::getOperationName().str()};

    std::stringstream out;
    std::stringstream err;
    int status;
    // TODO If we had arguments like `--no-columnar`, we could avoid the if-then-else here and simply use sth like
    // `useColumnar ? "--columnar" : "--no-columnar"`.
    if (useColumnar)
        status = runDaphne(out, err, "--explain", "columnar", "--columnar", scriptFilePath.c_str());
    else
        status = runDaphne(out, err, "--explain", "columnar", scriptFilePath.c_str());

    CHECK(status == StatusCode::SUCCESS);
    CHECK(out.str() == exp);
    for (const std::string &opName : requiredOps)
        CHECK_THAT(err.str(), Catch::Contains(opName));
    for (const std::string &opName : disallowedOps)
        CHECK_THAT(err.str(), !Catch::Contains(opName));
    for (std::string &opName : cnvOps)
        CHECK_THAT(err.str(), !Catch::Contains(opName));
}

TEST_CASE("columnar", TAG_SQL) {
    // The operations to check for in the IR. The elements of this array correspond to the test cases columnar_*. Each
    // element is a pair of (1) the decisive frame operations that should be used when running the script without
    // --columnar, and (2) the decisive column operations that should be used when running the script with --columnar.
    std::pair<std::vector<std::string> /*frmOps*/, std::vector<std::string> /*colOps*/> expectedOps[] = {
        // columnar_1
        {{}, {}},
        // columnar_2
        {{mlir::daphne::EwGeOp::getOperationName().str(), mlir::daphne::EwEqOp::getOperationName().str(),
          mlir::daphne::EwLeOp::getOperationName().str(), mlir::daphne::EwAndOp::getOperationName().str(),
          mlir::daphne::EwOrOp::getOperationName().str(), mlir::daphne::FilterRowOp::getOperationName().str()},
         {mlir::daphne::ColSelectGeOp::getOperationName().str(), mlir::daphne::ColSelectEqOp::getOperationName().str(),
          mlir::daphne::ColSelectLeOp::getOperationName().str(), mlir::daphne::ColIntersectOp::getOperationName().str(),
          mlir::daphne::ColMergeOp::getOperationName().str(), mlir::daphne::ColProjectOp::getOperationName().str()}},
        // columnar_3
        {{mlir::daphne::AllAggSumOp::getOperationName().str()},
         {mlir::daphne::ColAllAggSumOp::getOperationName().str()}},
        // columnar_4
        {{mlir::daphne::GroupOp::getOperationName().str()},
         {mlir::daphne::ColGroupFirstOp::getOperationName().str(),
          mlir::daphne::ColGrpAggSumOp::getOperationName().str()}},
        // columnar_5
        {{mlir::daphne::GroupOp::getOperationName().str()},
         {mlir::daphne::ColGroupFirstOp::getOperationName().str(),
          mlir::daphne::ColGroupNextOp::getOperationName().str(),
          mlir::daphne::ColGrpAggSumOp::getOperationName().str()}},
        // columnar_6
        {{mlir::daphne::EwEqOp::getOperationName().str(), mlir::daphne::EwLtOp::getOperationName().str(),
          mlir::daphne::FilterRowOp::getOperationName().str(), mlir::daphne::InnerJoinOp::getOperationName().str(),
          mlir::daphne::AllAggSumOp::getOperationName().str()},
         {mlir::daphne::ColSelectEqOp::getOperationName().str(), mlir::daphne::ColSelectLtOp::getOperationName().str(),
          mlir::daphne::ColJoinOp::getOperationName().str(), mlir::daphne::ColAllAggSumOp::getOperationName().str()}},
        // columnar_7
        // semi join is not generated by the SQL parser yet (it uses inner join instead)
        {{mlir::daphne::EwEqOp::getOperationName().str(), mlir::daphne::EwLtOp::getOperationName().str(),
          mlir::daphne::EwOrOp::getOperationName().str(),
          mlir::daphne::InnerJoinOp::getOperationName().str(), /*mlir::daphne::SemiJoinOp::getOperationName().str(),*/
          mlir::daphne::GroupOp::getOperationName().str()},
         {mlir::daphne::ColSelectEqOp::getOperationName().str(), mlir::daphne::ColSelectLtOp::getOperationName().str(),
          mlir::daphne::ColMergeOp::getOperationName().str(),
          mlir::daphne::ColJoinOp::getOperationName().str(), /*mlir::daphne::ColSemiJoinOp::getOperationName().str(),*/
          mlir::daphne::ColGroupFirstOp::getOperationName().str(),
          mlir::daphne::ColGrpAggSumOp::getOperationName().str()}}
        // end
    };

    // We have multiple test queries, each of which is expressed in both DaphneDSL and SQL.
    for (size_t i = 1; i <= 7; i++) {
        for (std::string lang : {"daphnedsl", "sql"}) {
            DYNAMIC_SECTION("columnar_" << i << "_" << lang << ".daphne") {
                const std::string scriptFilePath = dirPath + "columnar_" + std::to_string(i) + "_" + lang + ".daphne";

                // Read the expected result.
                const std::string refFilePath = dirPath + "columnar_" + std::to_string(i) + ".txt";
                const std::string exp = readTextFile(refFilePath);

                // Check if DAPHNE runs successfully and produces the correct result both without and with --columnar.
                // Here, we don't use --explain and expect cerr to be empty.
                compareDaphneToStr(exp, scriptFilePath);
                compareDaphneToStr(exp, scriptFilePath, "--columnar");

                // Check if DAPHNE uses the expected operations and doesn't use the unexpected operations to run the
                // query/script (frame ops without --columnar, column ops with --columnar).
                std::vector<std::string> frmOps = expectedOps[i - 1].first;
                std::vector<std::string> colOps = expectedOps[i - 1].second;
                checkOps(exp, scriptFilePath, false, frmOps, colOps);
                checkOps(exp, scriptFilePath, true, colOps, frmOps);
            }
        }
    }
}