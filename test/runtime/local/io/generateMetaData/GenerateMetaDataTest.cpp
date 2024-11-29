#include <catch.hpp>
#include <runtime/local/io/FileMetaData.h>
#include <runtime/local/io/utils.h>
#include <fstream>
#include <parser/metadata/MetaDataParser.h>
#include <iostream>

const std::string dirPath = "/daphne/test/runtime/local/io/generateMetaData/";

TEST_CASE("generated metadata saved correctly", "[metadata]") {
    std::string csvFilename = dirPath + "generateMetaData.csv";
    //saving generated metadata with first read
    FileMetaData generatedMetaData = MetaDataParser::readMetaData(csvFilename, true, true);
    //reading metadata from saved file
    FileMetaData readMD = MetaDataParser::readMetaData(csvFilename, true, true);

    REQUIRE(generatedMetaData.numCols == readMD.numCols);
    REQUIRE(generatedMetaData.numRows == readMD.numRows);
    REQUIRE(generatedMetaData.isSingleValueType == readMD.isSingleValueType);
    REQUIRE(generatedMetaData.schema == readMD.schema);
    REQUIRE(generatedMetaData.labels == readMD.labels);
    REQUIRE(std::filesystem::exists(csvFilename + ".meta"));
    std::filesystem::remove(csvFilename + ".meta");
}

TEST_CASE("generate meta data for frame with labels", "[metadata]") {
    std::string csvFilename = dirPath + "generateMetaData.csv";
    FileMetaData generatedMetaData = generateFileMetaData(csvFilename, true, true);
    REQUIRE(generatedMetaData.numRows == 3);
    REQUIRE(generatedMetaData.numCols == 3);
    REQUIRE(generatedMetaData.schema[0] == ValueTypeCode::SI8);
    REQUIRE(generatedMetaData.schema[1] == ValueTypeCode::SI8);
    REQUIRE(generatedMetaData.schema[2] == ValueTypeCode::SI8);
    REQUIRE(generatedMetaData.labels[0] == "label1");
    REQUIRE(generatedMetaData.labels[1] == "label2");
    REQUIRE(generatedMetaData.labels[2] == "label3");
}

TEST_CASE("generate meta data for frame with type uint64", "[metadata]"){
    std::string csvFilename = dirPath + "generateMetaData1.csv";
    FileMetaData generatedMetaData = generateFileMetaData(csvFilename, false, true);
    REQUIRE(generatedMetaData.numRows == 2);
    REQUIRE(generatedMetaData.numCols == 2);
    REQUIRE(generatedMetaData.schema[0] == ValueTypeCode::UI64);
    REQUIRE(generatedMetaData.schema[1] == ValueTypeCode::UI64);
}

TEST_CASE("generate meta data for matrix with type uint64", "[metadata]"){
    std::string csvFilename = dirPath + "generateMetaData1.csv";
    FileMetaData generatedMetaData = generateFileMetaData(csvFilename, false, false);
    REQUIRE(generatedMetaData.numRows == 2);
    REQUIRE(generatedMetaData.numCols == 2);
    REQUIRE(generatedMetaData.isSingleValueType == true);
    REQUIRE(generatedMetaData.schema[0] == ValueTypeCode::UI64);
}

TEST_CASE("generate meta data for frame with type int64", "[metadata]"){
    std::string csvFilename = dirPath + "generateMetaData2.csv";
    FileMetaData generatedMetaData = generateFileMetaData(csvFilename, false, true);
    REQUIRE(generatedMetaData.numRows == 2);
    REQUIRE(generatedMetaData.numCols == 2);
    REQUIRE(generatedMetaData.schema[0] == ValueTypeCode::SI64);
    REQUIRE(generatedMetaData.schema[1] == ValueTypeCode::SI64);
}

TEST_CASE("generate meta data for matrix with type int64", "[metadata]"){
    std::string csvFilename = dirPath + "generateMetaData2.csv";
    FileMetaData generatedMetaData = generateFileMetaData(csvFilename, false, false);
    REQUIRE(generatedMetaData.numRows == 2);
    REQUIRE(generatedMetaData.numCols == 2);
    REQUIRE(generatedMetaData.isSingleValueType == true);
    REQUIRE(generatedMetaData.schema[0] == ValueTypeCode::SI64);
}

TEST_CASE("generate meta data for frame with type uint32", "[metadata]"){
    std::string csvFilename = dirPath + "generateMetaData3.csv";
    FileMetaData generatedMetaData = generateFileMetaData(csvFilename, false, true);
    REQUIRE(generatedMetaData.numRows == 2);
    REQUIRE(generatedMetaData.numCols == 2);
    std::cout << "Float (32-bit) max value: " << std::numeric_limits<float>::max() << std::endl;
    std::cout << "Float (32-bit) min value: " << std::numeric_limits<float>::lowest() << std::endl;
    REQUIRE(generatedMetaData.schema[0] == ValueTypeCode::UI32);
    REQUIRE(generatedMetaData.schema[1] == ValueTypeCode::UI32);
}

TEST_CASE("generate meta data for matrix with type uint32", "[metadata]"){
    std::string csvFilename = dirPath + "generateMetaData3.csv";
    FileMetaData generatedMetaData = generateFileMetaData(csvFilename, false, false);
    REQUIRE(generatedMetaData.numRows == 2);
    REQUIRE(generatedMetaData.numCols == 2);
    REQUIRE(generatedMetaData.isSingleValueType == true);
    REQUIRE(generatedMetaData.schema[0] == ValueTypeCode::UI32);
}

TEST_CASE("generate meta data for frame with type int32", "[metadata]"){
    std::string csvFilename = dirPath + "generateMetaData4.csv";
    FileMetaData generatedMetaData = generateFileMetaData(csvFilename, false, true);
    REQUIRE(generatedMetaData.numRows == 2);
    REQUIRE(generatedMetaData.numCols == 2);
    REQUIRE(generatedMetaData.schema[0] == ValueTypeCode::SI32);
    REQUIRE(generatedMetaData.schema[1] == ValueTypeCode::SI32);
}

TEST_CASE("generate meta data for matrix with type int32", "[metadata]"){
    std::string csvFilename = dirPath + "generateMetaData4.csv";
    FileMetaData generatedMetaData = generateFileMetaData(csvFilename, false, false);
    REQUIRE(generatedMetaData.numRows == 2);
    REQUIRE(generatedMetaData.numCols == 2);
    REQUIRE(generatedMetaData.isSingleValueType == true);
    REQUIRE(generatedMetaData.schema[0] == ValueTypeCode::SI32);
}

TEST_CASE("generate meta data for frame with type uint8", "[metadata]"){
    std::string csvFilename = dirPath + "generateMetaData5.csv";
    FileMetaData generatedMetaData = generateFileMetaData(csvFilename, false, true);
    REQUIRE(generatedMetaData.numRows == 2);
    REQUIRE(generatedMetaData.numCols == 2);
    REQUIRE(generatedMetaData.schema[0] == ValueTypeCode::UI8);
    REQUIRE(generatedMetaData.schema[1] == ValueTypeCode::UI8);
}

TEST_CASE("generate meta data for matrix with type uint8", "[metadata]"){
    std::string csvFilename = dirPath + "generateMetaData5.csv";
    FileMetaData generatedMetaData = generateFileMetaData(csvFilename, false, false);
    REQUIRE(generatedMetaData.numRows == 2);
    REQUIRE(generatedMetaData.numCols == 2);
    REQUIRE(generatedMetaData.isSingleValueType == true);
    REQUIRE(generatedMetaData.schema[0] == ValueTypeCode::UI8);
}

TEST_CASE("generate meta data for frame with type int8", "[metadata]"){
    std::string csvFilename = dirPath + "generateMetaData6.csv";
    FileMetaData generatedMetaData = generateFileMetaData(csvFilename, false, true);
    REQUIRE(generatedMetaData.numRows == 2);
    REQUIRE(generatedMetaData.numCols == 3);
    REQUIRE(generatedMetaData.schema[0] == ValueTypeCode::SI8);
    REQUIRE(generatedMetaData.schema[1] == ValueTypeCode::SI8);
    REQUIRE(generatedMetaData.schema[2] == ValueTypeCode::SI8);
}

TEST_CASE("generate meta data for matrix with type int8", "[metadata]"){
    std::string csvFilename = dirPath + "generateMetaData6.csv";
    FileMetaData generatedMetaData = generateFileMetaData(csvFilename, false, false);
    REQUIRE(generatedMetaData.numRows == 2);
    REQUIRE(generatedMetaData.numCols == 3);
    REQUIRE(generatedMetaData.isSingleValueType == true);
    REQUIRE(generatedMetaData.schema[0] == ValueTypeCode::SI8);
}

TEST_CASE("generate meta data for frame with type float", "[metadata]"){
    std::string csvFilename = dirPath + "generateMetaData7.csv";
    FileMetaData generatedMetaData = generateFileMetaData(csvFilename, false, true);
    REQUIRE(generatedMetaData.numRows == 2);
    REQUIRE(generatedMetaData.numCols == 3);
    REQUIRE(generatedMetaData.schema[0] == ValueTypeCode::F32);
    REQUIRE(generatedMetaData.schema[1] == ValueTypeCode::F32);
    REQUIRE(generatedMetaData.schema[2] == ValueTypeCode::F32);
}

TEST_CASE("generate meta data for matrix with type float", "[metadata]"){
    std::string csvFilename = dirPath + "generateMetaData7.csv";
    FileMetaData generatedMetaData = generateFileMetaData(csvFilename, false, false);
    REQUIRE(generatedMetaData.numRows == 2);//TODO: look at precision
    REQUIRE(generatedMetaData.numCols == 3);
    REQUIRE(generatedMetaData.isSingleValueType == true);
    REQUIRE(generatedMetaData.schema[0] == ValueTypeCode::F32);
}

TEST_CASE("generate meta data for frame with type double", "[metadata]"){
    std::string csvFilename = dirPath + "generateMetaData8.csv";
    FileMetaData generatedMetaData = generateFileMetaData(csvFilename, false, true);
    REQUIRE(generatedMetaData.numRows == 2);
    REQUIRE(generatedMetaData.numCols == 2);
    REQUIRE(generatedMetaData.schema[0] == ValueTypeCode::F64);
    REQUIRE(generatedMetaData.schema[1] == ValueTypeCode::F64);
}

TEST_CASE("generate meta data for matrix with type double", "[metadata]"){
    std::string csvFilename = dirPath + "generateMetaData8.csv";
    FileMetaData generatedMetaData = generateFileMetaData(csvFilename, false, false);
    REQUIRE(generatedMetaData.numRows == 2);
    REQUIRE(generatedMetaData.numCols == 2);
    REQUIRE(generatedMetaData.isSingleValueType == true);
    REQUIRE(generatedMetaData.schema[0] == ValueTypeCode::F64);
}

TEST_CASE("generate meta data for frame with labels and mixed types", "[metadata]"){
    std::string csvFilename = dirPath + "generateMetaData9.csv";
    FileMetaData generatedMetaData = generateFileMetaData(csvFilename, true, true);
    REQUIRE(generatedMetaData.numRows == 2);
    REQUIRE(generatedMetaData.numCols == 5);
    REQUIRE(generatedMetaData.schema[0] == ValueTypeCode::SI8);
    REQUIRE(generatedMetaData.schema[1] == ValueTypeCode::FIXEDSTR16);
    REQUIRE(generatedMetaData.schema[2] == ValueTypeCode::STR);
    REQUIRE(generatedMetaData.schema[3] == ValueTypeCode::F32);
    REQUIRE(generatedMetaData.schema[4] == ValueTypeCode::SI32);
    REQUIRE(generatedMetaData.labels[0] == "label1");
    REQUIRE(generatedMetaData.labels[1] == "label2");
    REQUIRE(generatedMetaData.labels[2] == "label3");
    REQUIRE(generatedMetaData.labels[3] == "label4");
    REQUIRE(generatedMetaData.labels[4] == "\"label5\"");
}

TEST_CASE("generate meta data for frame with mixed types", "[metadata]"){
    std::string csvFilename = dirPath + "generateMetaData10.csv";
    FileMetaData generatedMetaData = generateFileMetaData(csvFilename, false, true);
    REQUIRE(generatedMetaData.numRows == 2);
    REQUIRE(generatedMetaData.numCols == 5);
    REQUIRE(generatedMetaData.isSingleValueType == false);
    REQUIRE(generatedMetaData.schema[0] == ValueTypeCode::SI8);
    REQUIRE(generatedMetaData.schema[1] == ValueTypeCode::FIXEDSTR16);
    REQUIRE(generatedMetaData.schema[2] == ValueTypeCode::STR);
    REQUIRE(generatedMetaData.schema[3] == ValueTypeCode::F32);
    REQUIRE(generatedMetaData.schema[4] == ValueTypeCode::SI32);
}

TEST_CASE("generate meta data for matrix with mixed types", "[metadata]"){
    std::string csvFilename = dirPath + "generateMetaData10.csv";
    FileMetaData generatedMetaData = generateFileMetaData(csvFilename, false, false);
    REQUIRE(generatedMetaData.numRows == 2);
    REQUIRE(generatedMetaData.numCols == 5);
    REQUIRE(generatedMetaData.isSingleValueType == true);
    REQUIRE(generatedMetaData.schema[0] == ValueTypeCode::STR);
}