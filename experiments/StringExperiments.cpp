#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Umbra.h>
#include <runtime/local/datastructures/UmbraNew.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>
#include <runtime/local/io/ReadCsv.h>
#include <runtime/local/kernels/EwBinaryMat.h>
#include <runtime/local/kernels/EwBinarySca.h>
#include <runtime/local/kernels/EwUnaryMat.h>
#include <runtime/local/kernels/Fill.h>
#include <runtime/local/kernels/OneHot.h>
#include <runtime/local/kernels/Reshape.h>
#include <runtime/local/kernels/Reverse.h>
#include <runtime/local/kernels/Transpose.h>

#include <tags.h>

#include <catch.hpp>

#include <cstdint>
#include <random>

#include <filesystem>
#include <vector>
#include <string>

#include <regex>
#include <unordered_map>

#define TEST_NAME(opName) "StringsExperiments (" opName ")"
#define CATCH_CONFIG_ENABLE_BENCHMARKING

#define LOOP_SIZE 1
#define NUM_COLS 2

#define DELIM ','

// Place this file into the test directory (./test/runtime/local/datastructures/), add it to the CMakeLists.txt in the test directory and perform and run ./experiments/experiments.sh -nb StringsE* -d yes
// The results.xml is then used by the analytics jupyter notebook

namespace fs = std::filesystem;

std::vector<std::string> getCSVFiles(const std::string& folderPath) {
    std::vector<std::string> csvFiles;
    for (const auto& entry : fs::directory_iterator(folderPath)) {
        if (entry.path().extension() == ".csv") {
            csvFiles.push_back(entry.path().string());
        }
    }
    return csvFiles;
}


namespace fs = std::filesystem;

std::vector<std::vector<std::string>> getCSVFilePairs(const std::string& folderPath) {
    // Map: key is the common filename (without the seed digit)
    // Value: pair.first is the seed 3 file, pair.second is the seed 1 file.
    std::unordered_map<std::string, std::pair<std::string, std::string>> fileMap;
    
    // This regex matches filenames with a substring "seed_" followed by either '1' or '3'
    // For example: "data_seed_1_version.csv" or "data_seed_3_version.csv"
    // Group 1: everything up to and including "seed_"
    // Group 2: the seed digit (either '1' or '3')
    // Group 3: the rest of the filename (excluding the ".csv" extension)
    std::regex pattern(R"(^(.*seed_)([13])(.*)\.csv$)", std::regex::icase);

    for (const auto& entry : fs::directory_iterator(folderPath)) {
        if (entry.path().extension() == ".csv") {
            std::string fileName = entry.path().filename().string();
            std::smatch match;
            if (std::regex_match(fileName, match, pattern)) {
                // Ensure we have the expected groups: entire match, then 3 groups.
                if (match.size() == 4) {
                    std::string prefix = match[1].str();       // e.g., "data_seed_"
                    std::string seedDigit = match[2].str();      // "1" or "3"
                    std::string suffix = match[3].str();         // e.g., "_version"
                    
                    // Create a key by removing the seed digit.
                    std::string commonKey = prefix + suffix;
                    
                    // Depending on the digit, store the file in the proper slot.
                    if (seedDigit == "3") {
                        fileMap[commonKey].first = entry.path().string();
                    } else if (seedDigit == "1") {
                        fileMap[commonKey].second = entry.path().string();
                    }
                }
            }
        }
    }

    // Build the vector of pairs. Only include pairs that have both seed files.
    std::vector<std::vector<std::string>> result;
    for (const auto& [key, pair] : fileMap) {
        if (!pair.first.empty() && !pair.second.empty()) {
            // Each inner vector has 2 entries: [seed3_file, seed1_file]
            result.push_back({ pair.first, pair.second });
        }
    }
    return result;
}

const std::vector<std::vector<std::string>>& getCachedCSVFilePairs() {
    static const std::vector<std::vector<std::string>> cachedPairs = getCSVFilePairs("./experiments/datasets/");
    return cachedPairs;
}


//-----------------------------------------------------------------
// A generic binary matrix test function (unchanged)
template <class DTArg, class DTRes>
void StringTestEwBinaryMat(BinaryOpCode opCode, const DTArg *lhs, const DTArg *rhs) {
    DTRes *res = nullptr;
    ewBinaryMat<DTRes, DTArg, DTArg>(opCode, res, lhs, rhs, nullptr);
    DataObjectFactory::destroy(res);
}

//-----------------------------------------------------------------
// Generic unary matrix test function
template <class DTArg, class DTRes>
void StringTestEwUnaryMat(UnaryOpCode opCode, const DTArg *arg) {
    DTRes *res = nullptr;
    ewUnaryMat<DTRes, DTArg>(opCode, res, arg, nullptr);
    DataObjectFactory::destroy(res);
}

//-----------------------------------------------------------------
// Test for concatenation
template <typename VT>
void StringTestConcat(VT lhs, VT rhs) {
    EwBinarySca<BinaryOpCode::CONCAT, VT, VT, VT>::apply(lhs, rhs, nullptr);
}


template <BinaryOpCode opCode, typename T>
void StringTestEwBinarySca(T lhs, T rhs, int64_t exp) {
    EwBinarySca<opCode, int64_t, T, T>::apply(lhs, rhs, nullptr);
}

//-----------------------------------------------------------------

TEMPLATE_PRODUCT_TEST_CASE(
    TEST_NAME("ReadCsv"), 
    TAG_DATASTRUCTURES, 
    (DenseMatrix),             
    (ALL_STRING_VALUE_TYPES)) {

    using DT = TestType;
    // Specify the folder where your CSV files are located.
    std::string folderPath = "./experiments/datasets/";

    // Dynamically get all CSV files in that folder.
    auto csvFiles = getCachedCSVFilePairs();

    // Use Catch2's generator to iterate through CSV file names.
    auto filePair = GENERATE_COPY(from_range(csvFiles.begin(), csvFiles.end()));

    const std::string& fileName1 = filePair[0];
    const std::string& fileName2 = filePair[1];

    auto numRows = GENERATE(100000, 1000000);

    // Build an outer section name that embeds the CSV file name.
    std::string outerSectionName =fileName1;

    SECTION(outerSectionName + "_numRowsRead_" + std::to_string(numRows)) {
        DT* m = nullptr;
        SECTION("readCsv()") {
                for (size_t i = 0; i < LOOP_SIZE; i++) {
                    
                    // Use .c_str() to convert std::string to const char*
                    readCsv(m, fileName1.c_str(), numRows, NUM_COLS, DELIM);
                }
            }
        DataObjectFactory::destroy(m);
    }
}



TEMPLATE_PRODUCT_TEST_CASE(
    TEST_NAME("get"),
    TAG_DATASTRUCTURES,
    (DenseMatrix),
    (ALL_STRING_VALUE_TYPES)
) {
    using DT = TestType;
    using VT = typename DT::VT;

    // Specify the folder where your CSV files are located.
    std::string folderPath = "./experiments/datasets/";

    // Dynamically get all CSV files in that folder.
    auto csvFiles = getCachedCSVFilePairs();

    // Use Catch2's generator to iterate through CSV file names.
    auto filePair = GENERATE_COPY(from_range(csvFiles.begin(), csvFiles.end()));

    const std::string& fileName1 = filePair[0];

    auto numRows = GENERATE(100000, 1000000);

    // Build an outer section name that embeds the CSV file name.
    std::string outerSectionName =fileName1;
    SECTION(outerSectionName + "_numRowsRead_" + std::to_string(numRows)) {
        DT* m = nullptr;
        // Use .c_str() to convert std::string to const char*
        readCsv(m, fileName1.c_str(), numRows, NUM_COLS, DELIM);

        SECTION("getNumRows()") {
            for (size_t i = 0; i < LOOP_SIZE; i++) {
                volatile size_t numRowsLhs = m->getNumRows();
            }
        }
        SECTION("getNumCols()") {
            for (size_t i = 0; i < LOOP_SIZE; i++) {
                volatile size_t numColsLhs = m->getNumCols();
            }
        }
        SECTION("getValues()") {
            for (size_t i = 0; i < LOOP_SIZE; i++) {
                volatile VT* values = m->getValues();
            }
        }

        DataObjectFactory::destroy(m);
    }
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("EwBinaryMat"), TAG_DATASTRUCTURES, (DenseMatrix),
                           (ALL_STRING_VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;
    using DTRes = DenseMatrix<int64_t>;


    // Specify the folder where your CSV files are located.
    std::string folderPath = "./experiments/datasets/";

    // Dynamically get all CSV files in that folder.
    auto csvFiles = getCachedCSVFilePairs();

    // Use Catch2's generator to iterate through CSV file names.
    auto filePair = GENERATE_COPY(from_range(csvFiles.begin(), csvFiles.end()));

    const std::string& fileName1 = filePair[0];
    const std::string& fileName2 = filePair[1];

    auto numRows = GENERATE(100000, 1000000);

    // Build an outer section name that embeds the CSV file name.
    std::string outerSectionName = fileName1;

    SECTION(outerSectionName + "_numRowsRead_" + std::to_string(numRows)) {
        
        DT *m1 = nullptr;
        DT *m2 = nullptr;

        readCsv(m1, fileName1.c_str(), numRows, NUM_COLS, DELIM);
        readCsv(m2, fileName2.c_str(), numRows, NUM_COLS, DELIM);

        REQUIRE(m1->getNumRows() == numRows);
        REQUIRE(m1->getNumCols() == NUM_COLS);
        REQUIRE(m2->getNumRows() == numRows);
        REQUIRE(m2->getNumCols() == NUM_COLS);

        SECTION("EQ") {
            for (size_t i = 0; i < LOOP_SIZE; i++)
                StringTestEwBinaryMat<DT, DTRes>(BinaryOpCode::EQ, m1, m2);
        }

        SECTION("NEQ") {
            for (size_t i = 0; i < LOOP_SIZE; i++)
                StringTestEwBinaryMat<DT, DTRes>(BinaryOpCode::NEQ, m1, m2);
        }

        SECTION("LT") {
            for (size_t i = 0; i < LOOP_SIZE; i++)
                StringTestEwBinaryMat<DT, DTRes>(BinaryOpCode::LT, m1, m2);
        }

        SECTION("GT") {
            for (size_t i = 0; i < LOOP_SIZE; i++)
                StringTestEwBinaryMat<DT, DTRes>(BinaryOpCode::GT, m1, m2);
        }

        DataObjectFactory::destroy(m1);
        DataObjectFactory::destroy(m2);

    }    
}

    
TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("Operations"), TAG_DATASTRUCTURES, (DenseMatrix),
                           (ALL_STRING_VALUE_TYPES)) {
    using DT = TestType;

    // Specify the folder where your CSV files are located.
    std::string folderPath = "./experiments/datasets/";

    // Dynamically get all CSV files in that folder.
    auto csvFiles = getCachedCSVFilePairs();

    // Use Catch2's generator to iterate through CSV file names.
    auto filePair = GENERATE_COPY(from_range(csvFiles.begin(), csvFiles.end()));

    const std::string& fileName1 = filePair[0];

    auto numRows = GENERATE(100000, 1000000);

    // Build an outer section name that embeds the CSV file name.
    std::string outerSectionName = fileName1;

    SECTION(outerSectionName + "_numRowsRead_" + std::to_string(numRows)) {
        
        DT *m = nullptr;
        
        readCsv(m, fileName1.c_str(), numRows, NUM_COLS, DELIM);
        
        SECTION("Upper") {
            for (size_t i = 0; i < LOOP_SIZE; i++)
                StringTestEwUnaryMat<DT, DT>(UnaryOpCode::UPPER, m);
        }
    
        SECTION("Lower") {
            for (size_t i = 0; i < LOOP_SIZE; i++)
                StringTestEwUnaryMat<DT, DT>(UnaryOpCode::LOWER, m);
        }
    
        DataObjectFactory::destroy(m);
    }
}



TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("Operations2"), TAG_DATASTRUCTURES, (DenseMatrix),
                           (ALL_STRING_VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;


    // Specify the folder where your CSV files are located.
    std::string folderPath = "./experiments/datasets/";

    // Dynamically get all CSV files in that folder.
    auto csvFiles = getCachedCSVFilePairs();

    // Use Catch2's generator to iterate through CSV file names.
    auto filePair = GENERATE_COPY(from_range(csvFiles.begin(), csvFiles.end()));

    const std::string& fileName1 = filePair[0];

    auto numRows = GENERATE(100000, 1000000);

    // Build an outer section name that embeds the CSV file name.
    std::string outerSectionName = fileName1;

    SECTION(outerSectionName + "_numRowsRead_" + std::to_string(numRows)) {
        DT *m = nullptr;
        readCsv(m, fileName1.c_str(), numRows, NUM_COLS, DELIM);
        SECTION("Concat") {
            for (size_t r = 0; r < static_cast<size_t>(numRows); r++) {
                VT resultConcat = ewBinarySca<VT, VT, VT>(BinaryOpCode::CONCAT, m->get(r, 0), m->get(r, 1), nullptr);
            }
        }
        DataObjectFactory::destroy(m);
    }
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("Fill"), TAG_DATASTRUCTURES, (DenseMatrix),
                           (ALL_STRING_VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto numRows = GENERATE(100000, 1000000);

    DT *m = nullptr;
    
    SECTION("FillShortString") {
        fill(m, VT("abc"), numRows, NUM_COLS, nullptr);
    }

    SECTION("FillLongString") {
        fill(m, VT("abcdfgfdgfdgdfgfdgfdgfdggdfgfdgfdgfdjhg34hg3kjghj43hgkj34hgjk34hjkgh34kjgh3j4hgj34hg3jkgh34jkhgj34hjkg34hggdfgfdgdfg"), numRows, NUM_COLS, nullptr);
    }
    DataObjectFactory::destroy(m);
    
}


TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("Transpose"), TAG_DATASTRUCTURES, (DenseMatrix),
                           (ALL_STRING_VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;


    // Specify the folder where your CSV files are located.
    std::string folderPath = "./experiments/datasets/";

    // Dynamically get all CSV files in that folder.
    auto csvFiles = getCachedCSVFilePairs();

    // Use Catch2's generator to iterate through CSV file names.
    auto filePair = GENERATE_COPY(from_range(csvFiles.begin(), csvFiles.end()));

    const std::string& fileName1 = filePair[0];

    auto numRows = GENERATE(100000, 1000000);

    // Build an outer section name that embeds the CSV file name.
    std::string outerSectionName = fileName1;

    SECTION(outerSectionName + "_numRowsRead_" + std::to_string(numRows)) {
        DT *m = nullptr;
        readCsv(m, fileName1.c_str(), numRows, NUM_COLS, DELIM);

        DT *res = nullptr;
        SECTION("Transpose") {
            transpose<DT, DT>(res, m, nullptr); 
        }
        DataObjectFactory::destroy(m, res);
    }
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("Reverse"), TAG_DATASTRUCTURES, (DenseMatrix),
                           (ALL_STRING_VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;


    // Specify the folder where your CSV files are located.
    std::string folderPath = "./experiments/datasets/";

    // Dynamically get all CSV files in that folder.
    auto csvFiles = getCachedCSVFilePairs();

    // Use Catch2's generator to iterate through CSV file names.
    auto filePair = GENERATE_COPY(from_range(csvFiles.begin(), csvFiles.end()));

    const std::string& fileName1 = filePair[0];

    auto numRows = GENERATE(100000, 1000000);

    // Build an outer section name that embeds the CSV file name.
    std::string outerSectionName = fileName1;

    SECTION(outerSectionName + "_numRowsRead_" + std::to_string(numRows)) {
        DT *m = nullptr;
        readCsv(m, fileName1.c_str(), numRows, NUM_COLS, DELIM);

        DT *res = nullptr;
        SECTION("Reverse") {
            reverse<DT, DT>(res, m, nullptr); 
        }
        DataObjectFactory::destroy(m, res);
    }
}


TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("Reshape"), TAG_DATASTRUCTURES, (DenseMatrix),
                           (ALL_STRING_VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;


    // Specify the folder where your CSV files are located.
    std::string folderPath = "./experiments/datasets/";

    // Dynamically get all CSV files in that folder.
    auto csvFiles = getCachedCSVFilePairs();

    // Use Catch2's generator to iterate through CSV file names.
    auto filePair = GENERATE_COPY(from_range(csvFiles.begin(), csvFiles.end()));

    const std::string& fileName1 = filePair[0];

    auto numRows = GENERATE(100000, 1000000);

    // Build an outer section name that embeds the CSV file name.
    std::string outerSectionName = fileName1;

    SECTION(outerSectionName + "_numRowsRead_" + std::to_string(numRows)) {
        DT *m = nullptr;
        readCsv(m, fileName1.c_str(), numRows, NUM_COLS, DELIM);

        DT *res = nullptr;
        SECTION("Reshape") {
            reshape<DT, DT>(res, m, NUM_COLS, numRows, nullptr);
        }
        DataObjectFactory::destroy(m, res);
    }
}