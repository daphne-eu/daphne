#include <catch.hpp>
#include <fstream>
#include <iostream>
#include <parser/metadata/MetaDataParser.h>
#include <runtime/local/io/FileMetaData.h>
#include <runtime/local/io/utils.h>
#include <utility>

const std::string dirPath = "/daphne/test/runtime/local/io/generateMetaData/";

class FileCleanupFixture {
  public:
    std::string fileName;

    explicit FileCleanupFixture(std::string filename) : fileName(std::move(filename)) { cleanup(); }

    ~FileCleanupFixture() { cleanup(); }

  private:
    void cleanup() const {
        if (std::filesystem::exists(fileName + ".meta")) {
            std::filesystem::remove(fileName + ".meta");
        }
    }
};

TEST_CASE("generated metadata saved correctly", "[metadata]") {
    std::string csvFilename = dirPath + "generateMetaData.csv";
    FileCleanupFixture cleanup(csvFilename); // cleans up before and after the test
    // saving generated metadata with first read
    FileMetaData generatedMetaData = MetaDataParser::readMetaData(csvFilename, ',');
    // reading metadata from saved file
    FileMetaData readMD = MetaDataParser::readMetaData(csvFilename, ',');

    REQUIRE(generatedMetaData.numCols == readMD.numCols);
    REQUIRE(generatedMetaData.numRows == readMD.numRows);
    REQUIRE(generatedMetaData.isSingleValueType == readMD.isSingleValueType);
    REQUIRE(generatedMetaData.schema == readMD.schema);
    REQUIRE(generatedMetaData.labels == readMD.labels);
    REQUIRE(std::filesystem::exists(csvFilename + ".meta"));
}

TEST_CASE("generated metadata saved correctly for frame with single value type", "[metadata]") {
    std::string csvFilename = dirPath + "generateMetaDataSingleValue.csv";
    FileCleanupFixture cleanup(csvFilename); // cleans up before and after the test
    // saving generated metadata with first read
    FileMetaData generatedMetaData = MetaDataParser::readMetaData(csvFilename, ',');
    // reading metadata from saved file
    FileMetaData readMD = MetaDataParser::readMetaData(csvFilename, ',');

    REQUIRE(generatedMetaData.numCols == readMD.numCols);
    REQUIRE(generatedMetaData.numRows == readMD.numRows);
    REQUIRE(generatedMetaData.isSingleValueType == readMD.isSingleValueType);
    REQUIRE(generatedMetaData.schema == readMD.schema);
    REQUIRE(generatedMetaData.labels == readMD.labels);
    REQUIRE(std::filesystem::exists(csvFilename + ".meta"));
}

TEST_CASE("generated metadata saved correctly for matrix with single value type", "[metadata]") {
    std::string csvFilename = dirPath + "generateMetaDataSingleValue.csv";
    FileCleanupFixture cleanup(csvFilename); // cleans up before and after the test
    // saving generated metadata with first read
    FileMetaData generatedMetaData = MetaDataParser::readMetaData(csvFilename, ',', true);
    // reading metadata from saved file
    FileMetaData readMD = MetaDataParser::readMetaData(csvFilename, ',', true);

    REQUIRE(generatedMetaData.numCols == readMD.numCols);
    REQUIRE(generatedMetaData.numRows == readMD.numRows);
    REQUIRE(generatedMetaData.isSingleValueType == readMD.isSingleValueType);
    REQUIRE(generatedMetaData.schema == readMD.schema);
    REQUIRE(generatedMetaData.labels == readMD.labels);
    REQUIRE(std::filesystem::exists(csvFilename + ".meta"));
}

TEST_CASE("generate meta data for frame", "[metadata]") {
    std::string csvFilename = dirPath + "generateMetaData.csv";
    FileCleanupFixture cleanup(csvFilename); // cleans up before and after the test
    FileMetaData generatedMetaData = generateFileMetaData(csvFilename, ',', 3);
    REQUIRE(generatedMetaData.numRows == 3);
    REQUIRE(generatedMetaData.numCols == 3);
    REQUIRE(generatedMetaData.isSingleValueType == false);
    REQUIRE(generatedMetaData.schema[0] == ValueTypeCode::SI8);
    REQUIRE(generatedMetaData.schema[1] == ValueTypeCode::SI8);
    REQUIRE(generatedMetaData.schema[2] == ValueTypeCode::UI8);
    for (int i = 0; i < 3; i++) {
        REQUIRE(generatedMetaData.labels[i] == "col_" + std::to_string(i));
    }
}

TEST_CASE("generate meta data for frame with type uint64", "[metadata]") {
    std::string csvFilename = dirPath + "generateMetaData1.csv";
    FileCleanupFixture cleanup(csvFilename); // cleans up before and after the test
    FileMetaData generatedMetaData = generateFileMetaData(csvFilename, ',', 2);
    REQUIRE(generatedMetaData.numRows == 2);
    REQUIRE(generatedMetaData.numCols == 2);
    REQUIRE(generatedMetaData.isSingleValueType == true);
    REQUIRE(generatedMetaData.schema[0] == ValueTypeCode::UI64);
}

TEST_CASE("generate meta data for matrix with type uint64", "[metadata]") {
    std::string csvFilename = dirPath + "generateMetaData1.csv";
    FileCleanupFixture cleanup(csvFilename); // cleans up before and after the test
    FileMetaData generatedMetaData = generateFileMetaData(csvFilename, ',', 2, true);
    REQUIRE(generatedMetaData.numRows == 2);
    REQUIRE(generatedMetaData.numCols == 2);
    REQUIRE(generatedMetaData.isSingleValueType == true);
    REQUIRE(generatedMetaData.schema[0] == ValueTypeCode::UI64);
}

TEST_CASE("generate meta data for frame with type int64", "[metadata]") {
    std::string csvFilename = dirPath + "generateMetaData2.csv";
    FileCleanupFixture cleanup(csvFilename); // cleans up before and after the test
    FileMetaData generatedMetaData = generateFileMetaData(csvFilename, ',', 2);
    REQUIRE(generatedMetaData.numRows == 2);
    REQUIRE(generatedMetaData.numCols == 2);
    REQUIRE(generatedMetaData.isSingleValueType == true);
    REQUIRE(generatedMetaData.schema[0] == ValueTypeCode::SI64);
}

TEST_CASE("generate meta data for matrix with type int64", "[metadata]") {
    std::string csvFilename = dirPath + "generateMetaData2.csv";
    FileCleanupFixture cleanup(csvFilename); // cleans up before and after the test
    FileMetaData generatedMetaData = generateFileMetaData(csvFilename, ',', 2, true);
    REQUIRE(generatedMetaData.numRows == 2);
    REQUIRE(generatedMetaData.numCols == 2);
    REQUIRE(generatedMetaData.isSingleValueType == true);
    REQUIRE(generatedMetaData.schema[0] == ValueTypeCode::SI64);
}

TEST_CASE("generate meta data for frame with type uint32", "[metadata]") {
    std::string csvFilename = dirPath + "generateMetaData3.csv";
    FileCleanupFixture cleanup(csvFilename); // cleans up before and after the test
    FileMetaData generatedMetaData = generateFileMetaData(csvFilename, ',', 2);
    REQUIRE(generatedMetaData.numRows == 2);
    REQUIRE(generatedMetaData.numCols == 2);
    REQUIRE(generatedMetaData.isSingleValueType == true);
    REQUIRE(generatedMetaData.schema[0] == ValueTypeCode::UI32);
}

TEST_CASE("generate meta data for matrix with type uint32", "[metadata]") {
    std::string csvFilename = dirPath + "generateMetaData3.csv";
    FileCleanupFixture cleanup(csvFilename); // cleans up before and after the test
    FileMetaData generatedMetaData = generateFileMetaData(csvFilename, ',', 2, true);
    REQUIRE(generatedMetaData.numRows == 2);
    REQUIRE(generatedMetaData.numCols == 2);
    REQUIRE(generatedMetaData.isSingleValueType == true);
    REQUIRE(generatedMetaData.schema[0] == ValueTypeCode::UI32);
}

TEST_CASE("generate meta data for frame with type int32", "[metadata]") {
    std::string csvFilename = dirPath + "generateMetaData4.csv";
    FileCleanupFixture cleanup(csvFilename); // cleans up before and after the test
    FileMetaData generatedMetaData = generateFileMetaData(csvFilename, ',', 2);
    REQUIRE(generatedMetaData.numRows == 2);
    REQUIRE(generatedMetaData.numCols == 2);
    REQUIRE(generatedMetaData.isSingleValueType == true);
    REQUIRE(generatedMetaData.schema[0] == ValueTypeCode::SI32);
}

TEST_CASE("generate meta data for matrix with type int32", "[metadata]") {
    std::string csvFilename = dirPath + "generateMetaData4.csv";
    FileCleanupFixture cleanup(csvFilename); // cleans up before and after the test
    FileMetaData generatedMetaData = generateFileMetaData(csvFilename, ',', 2, true);
    REQUIRE(generatedMetaData.numRows == 2);
    REQUIRE(generatedMetaData.numCols == 2);
    REQUIRE(generatedMetaData.isSingleValueType == true);
    REQUIRE(generatedMetaData.schema[0] == ValueTypeCode::SI32);
}

TEST_CASE("generate meta data for frame with type uint8", "[metadata]") {
    std::string csvFilename = dirPath + "generateMetaData5.csv";
    FileCleanupFixture cleanup(csvFilename); // cleans up before and after the test
    FileMetaData generatedMetaData = generateFileMetaData(csvFilename, ',', 2);
    REQUIRE(generatedMetaData.numRows == 2);
    REQUIRE(generatedMetaData.numCols == 3);
    REQUIRE(generatedMetaData.isSingleValueType == false);
    REQUIRE(generatedMetaData.schema[0] == ValueTypeCode::UI8);
    REQUIRE(generatedMetaData.schema[1] == ValueTypeCode::UI8);
    REQUIRE(generatedMetaData.schema[2] == ValueTypeCode::STR);
}

TEST_CASE("generate meta data for matrix with type uint8", "[metadata]") {
    std::string csvFilename = dirPath + "generateMetaData5_matrix.csv";
    FileCleanupFixture cleanup(csvFilename); // cleans up before and after the test
    FileMetaData generatedMetaData = generateFileMetaData(csvFilename, ',', 2, true);
    REQUIRE(generatedMetaData.numRows == 2);
    REQUIRE(generatedMetaData.numCols == 2);
    REQUIRE(generatedMetaData.isSingleValueType == true);
    REQUIRE(generatedMetaData.schema[0] == ValueTypeCode::UI8);
}

TEST_CASE("generate meta data for frame with type int8", "[metadata]") {
    std::string csvFilename = dirPath + "generateMetaData6.csv";
    FileCleanupFixture cleanup(csvFilename); // cleans up before and after the test
    FileMetaData generatedMetaData = generateFileMetaData(csvFilename, ',', 2);
    REQUIRE(generatedMetaData.numRows == 2);
    REQUIRE(generatedMetaData.numCols == 3);
    REQUIRE(generatedMetaData.isSingleValueType == true);
    REQUIRE(generatedMetaData.schema[0] == ValueTypeCode::SI8);
}

TEST_CASE("generate meta data for matrix with type int8", "[metadata]") {
    std::string csvFilename = dirPath + "generateMetaData6.csv";
    FileCleanupFixture cleanup(csvFilename); // cleans up before and after the test
    FileMetaData generatedMetaData = generateFileMetaData(csvFilename, ',', 2, true);
    REQUIRE(generatedMetaData.numRows == 2);
    REQUIRE(generatedMetaData.numCols == 3);
    REQUIRE(generatedMetaData.isSingleValueType == true);
    REQUIRE(generatedMetaData.schema[0] == ValueTypeCode::SI8);
}

TEST_CASE("generate meta data for frame with type float", "[metadata]") {
    std::string csvFilename = dirPath + "generateMetaData7.csv";
    FileCleanupFixture cleanup(csvFilename); // cleans up before and after the test
    FileMetaData generatedMetaData = generateFileMetaData(csvFilename, ',', 2);
    REQUIRE(generatedMetaData.numRows == 2);
    REQUIRE(generatedMetaData.numCols == 4);
    REQUIRE(generatedMetaData.isSingleValueType == false);
    REQUIRE(generatedMetaData.schema[0] == ValueTypeCode::F32);
    REQUIRE(generatedMetaData.schema[1] == ValueTypeCode::F32);
    REQUIRE(generatedMetaData.schema[2] == ValueTypeCode::F32);
    REQUIRE(generatedMetaData.schema[3] == ValueTypeCode::STR);
}

TEST_CASE("generate meta data for matrix with type float", "[metadata]") {
    std::string csvFilename = dirPath + "generateMetaData7_matrix.csv";
    FileCleanupFixture cleanup(csvFilename); // cleans up before and after the test
    FileMetaData generatedMetaData = generateFileMetaData(csvFilename, ',', 2, true);
    REQUIRE(generatedMetaData.numRows == 2);
    REQUIRE(generatedMetaData.numCols == 3);
    REQUIRE(generatedMetaData.isSingleValueType == true);
    REQUIRE(generatedMetaData.schema[0] == ValueTypeCode::F32);
}

TEST_CASE("generate meta data for frame with type double", "[metadata]") {
    std::string csvFilename = dirPath + "generateMetaData8.csv";
    FileCleanupFixture cleanup(csvFilename); // cleans up before and after the test
    FileMetaData generatedMetaData = generateFileMetaData(csvFilename, ',', 2);
    REQUIRE(generatedMetaData.numRows == 2);
    REQUIRE(generatedMetaData.numCols == 2);
    REQUIRE(generatedMetaData.isSingleValueType == true);
    REQUIRE(generatedMetaData.schema[0] == ValueTypeCode::F64);
}

TEST_CASE("generate meta data for matrix with type double", "[metadata]") {
    std::string csvFilename = dirPath + "generateMetaData8.csv";
    FileCleanupFixture cleanup(csvFilename); // cleans up before and after the test
    FileMetaData generatedMetaData = generateFileMetaData(csvFilename, ',', 2, true);
    REQUIRE(generatedMetaData.numRows == 2);
    REQUIRE(generatedMetaData.numCols == 2);
    REQUIRE(generatedMetaData.isSingleValueType == true);
    REQUIRE(generatedMetaData.schema[0] == ValueTypeCode::F64);
}

TEST_CASE("generate meta data for frame with mixed types", "[metadata]") {
    std::string csvFilename = dirPath + "generateMetaData9.csv";
    FileCleanupFixture cleanup(csvFilename); // cleans up before and after the test
    FileMetaData generatedMetaData = generateFileMetaData(csvFilename, ',', 2);
    REQUIRE(generatedMetaData.numRows == 2);
    REQUIRE(generatedMetaData.numCols == 6);
    REQUIRE(generatedMetaData.isSingleValueType == false);
    REQUIRE(generatedMetaData.schema[0] == ValueTypeCode::SI8);
    REQUIRE(generatedMetaData.schema[1] == ValueTypeCode::FIXEDSTR16);
    REQUIRE(generatedMetaData.schema[2] == ValueTypeCode::STR);
    REQUIRE(generatedMetaData.schema[3] == ValueTypeCode::F32);
    REQUIRE(generatedMetaData.schema[4] == ValueTypeCode::SI32);
    REQUIRE(generatedMetaData.schema[5] == ValueTypeCode::STR);
    for (int i = 0; i < 5; i++) {
        REQUIRE(generatedMetaData.labels[i] == "col_" + std::to_string(i));
    }
}

TEST_CASE("generate meta data for matrix with mixed types", "[metadata]") {
    std::string csvFilename = dirPath + "generateMetaData9.csv";
    FileCleanupFixture cleanup(csvFilename); // cleans up before and after the test
    FileMetaData generatedMetaData = generateFileMetaData(csvFilename, ',', 2, true);
    REQUIRE(generatedMetaData.numRows == 2);
    REQUIRE(generatedMetaData.numCols == 6);
    REQUIRE(generatedMetaData.isSingleValueType == true);
    REQUIRE(generatedMetaData.schema[0] == ValueTypeCode::STR);
}