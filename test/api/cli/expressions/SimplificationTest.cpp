#include <api/cli/Utils.h>

#include <tags.h>

#include <catch.hpp>

#include <sstream>
#include <string>

const std::string dirPath = "test/api/cli/expressions/";

#define MAKE_TEST_CASE(name, count)                                                                                    \
    TEST_CASE(name, TAG_REWRITE) {                                                                                     \
        for (unsigned i = 1; i <= count; i++) {                                                                        \
            DYNAMIC_SECTION(name "_" << i << ".daphne") { compareDaphneToRefSimple(dirPath, name, i); }                \
        }                                                                                                              \
    }

MAKE_TEST_CASE("simplf_sumEwadd", 1)
MAKE_TEST_CASE("simplf_sumTranspose", 1)
MAKE_TEST_CASE("simplf_sumMulLambda", 1)
MAKE_TEST_CASE("simplf_sumTrace", 1)
MAKE_TEST_CASE("simplf_mmSlice", 1)
MAKE_TEST_CASE("simplf_dynInsert", 1)
