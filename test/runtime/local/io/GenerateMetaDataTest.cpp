#include <catch.hpp>
#include <runtime/local/io/FileMetaData.h>
#include <runtime/local/io/utils.h>
#include <fstream>
#include <sstream>
#include <parser/metadata/MetaDataParser.h>

const std::string dirPath = "test/runtime/local/io/";

TEST_CASE("generated metadata matches saved metadata", "[metadata]") {
    for (int i = 1; i <= 5; ++i) {
        std::string rootPath = "\\\\wsl.localhost\\Ubuntu-CUDA\\home\\projects\\daphne\\test\\runtime\\local\\io\\";
        std::string csvFilename = dirPath + "ReadCsv" + std::to_string(i) + ".csv";

        // Read metadata from saved metadata file
        FileMetaData readMetaData = MetaDataParser::readMetaData(csvFilename);

        // Generate metadata from CSV file
        FileMetaData generatedMetaData = generateFileMetaData(csvFilename);

        // Check if the generated metadata matches the read metadata
        REQUIRE(generatedMetaData.numRows == readMetaData.numRows);
        REQUIRE(generatedMetaData.numCols == readMetaData.numCols);
        REQUIRE(generatedMetaData.isSingleValueType == readMetaData.isSingleValueType);
        REQUIRE(generatedMetaData.schema == readMetaData.schema);
        REQUIRE(generatedMetaData.labels == readMetaData.labels);
    }
}