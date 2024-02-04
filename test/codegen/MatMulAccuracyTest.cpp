#include <api/cli/Utils.h>
#include <limits>
#include <tags.h>

#include <catch.hpp>
#include <sstream>
#include <string>

#include "api/cli/StatusCode.h"

const std::string dirPath = "test/api/cli/codegen/";

TEST_CASE("matmul accuracy", "[codegen][matmul]") {
std::string result = readTextFile(dirPath + "matmul128.result");
    double epsilon = std::numeric_limits<double>().epsilon();
    compareDaphneToStringNumerically(result, dirPath + "matmul128.daphne", 1, epsilon);
    compareDaphneToStringNumerically(result, dirPath + "matmul128.daphne", 1, epsilon, "--mlir-codegen");
    compareDaphneToStringNumerically(result, dirPath + "matmul128.daphne", 1, epsilon, "--mlir-codegen", "--matmul-fixed-tile-sizes=2,3,4,5,6");
    compareDaphneToStringNumerically(result, dirPath + "matmul128.daphne", 1, epsilon, "--mlir-codegen", "--matmul-fixed-tile-sizes=2,3,4,5");
    compareDaphneToStringNumerically(result, dirPath + "matmul128.daphne", 1, epsilon, "--mlir-codegen", "--matmul-fixed-tile-sizes=2,3,4");
    compareDaphneToStringNumerically(result, dirPath + "matmul128.daphne", 1, epsilon, "--mlir-codegen", "--matmul-fixed-tile-sizes=2,3");
    compareDaphneToStringNumerically(result, dirPath + "matmul128.daphne", 1, epsilon, "--mlir-codegen", "--matmul-fixed-tile-sizes=2");
    compareDaphneToStringNumerically(result, dirPath + "matmul128.daphne", 1, epsilon, "--mlir-codegen", "--matmul-vec-size-bits=64");
    compareDaphneToStringNumerically(result, dirPath + "matmul128.daphne", 1, epsilon, "--mlir-codegen", "--matmul-fixed-tile-sizes=2,3,4", "--matmul-vec-size-bits=64");
}