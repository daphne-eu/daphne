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
#include <parser/metadata/MetaDataParser.h>
#include <tags.h>

#include <catch.hpp>

#include <fstream>
#include <regex>
#include <string>

std::string createDaphneScript(const std::string &evaluationDir,
                        const std::string &csvFilename,
                        const std::string &daphneScript) {
    std::filesystem::create_directories(evaluationDir); // ensure directory exists
    std::string daphneFilePath = evaluationDir + daphneScript;
    if (std::filesystem::exists(daphneFilePath)) {
        return daphneFilePath;
    }
    std::ofstream ofs(daphneFilePath);
    if(!ofs) {
        throw std::runtime_error("Could not create Daphne script file: " + daphneFilePath);
    }
    ofs << "readFrame(\"" << evaluationDir + csvFilename << "\");";
    ofs.close();
    return daphneFilePath;
}

template <typename... Args>
std::string runDaphneEval( const std::string &scriptFilePath, Args... args) {
    std::stringstream out;
    std::stringstream err;
    int status = runDaphne(out, err, args..., scriptFilePath.c_str());
    
    // Just CHECK (don't REQUIRE) success, such that in case of a failure, the
    // checks of out and err still run and provide useful messages. For err,
    // don't check empty(), because then catch2 doesn't display the error
    // output.
    CHECK(status == StatusCode::SUCCESS);
    //std::cout << out.str() << std::endl;
    //CHECK(err.str() == "");
    return out.str()+err.str();
}
// New data structure for timing values.
struct TimingData {
    // read time in nanoseconds as string (without the "ns" suffix)
    std::string readTime;
    std::string writeTime;
    double startupSeconds = 0.0;
    double parsingSeconds = 0.0;
    double compilationSeconds = 0.0;
    double executionSeconds = 0.0;
    double totalSeconds = 0.0;
};

// This function extracts timing information from the output string.
// Expected output format:
//   read time: 117784ns
//   {"startup_seconds": 0.0136333, "parsing_seconds": 0.000770869, "compilation_seconds": 0.0182154, "execution_seconds": 0.00726858, "total_seconds": 0.0398881}
TimingData extractTiming(const std::string &output, bool expectWriteTime = false) {
    TimingData timingData;
    std::istringstream iss(output);
    std::string line;
    // First line: read time.
    if(std::getline(iss, line)) {
        auto pos = line.find(":");
        if(pos != std::string::npos) {
            std::string val = line.substr(pos+1);
            // Trim leading spaces.
            while(!val.empty() && std::isspace(val.front())) {
                val.erase(val.begin());
            }
            // Remove "ns" suffix if present.
            if(val.size() >= 2 && val.substr(val.size()-2) == "ns") {
                val = val.substr(0, val.size()-2);
            }
            timingData.readTime = val;
        }
    }
    // Second line has write time
    if (expectWriteTime){
        if(std::getline(iss, line)) {
            auto pos = line.find(":");
            if(pos != std::string::npos) {
                std::string val = line.substr(pos+1);
                // Trim leading spaces.
                while(!val.empty() && std::isspace(val.front())) {
                    val.erase(val.begin());
                }
                // Remove "ns" suffix if present.
                if(val.size() >= 2 && val.substr(val.size()-2) == "ns") {
                    val = val.substr(0, val.size()-2);
                }
                timingData.writeTime = val;
            }
        }
    }
    
    // Second line: JSON with detailed timings.
    if(std::getline(iss, line)) {
        std::smatch match;
        std::regex regex_startup("\"startup_seconds\"\\s*:\\s*([0-9]*\\.?[0-9]+)");
        if(std::regex_search(line, match, regex_startup)) {
            timingData.startupSeconds = std::stod(match[1].str());
        }
        std::regex regex_parsing("\"parsing_seconds\"\\s*:\\s*([0-9]*\\.?[0-9]+)");
        if(std::regex_search(line, match, regex_parsing)) {
            timingData.parsingSeconds = std::stod(match[1].str());
        }
        std::regex regex_compilation("\"compilation_seconds\"\\s*:\\s*([0-9]*\\.?[0-9]+)");
        if(std::regex_search(line, match, regex_compilation)) {
            timingData.compilationSeconds = std::stod(match[1].str());
        }
        std::regex regex_execution("\"execution_seconds\"\\s*:\\s*([0-9]*\\.?[0-9]+)");
        if(std::regex_search(line, match, regex_execution)) {
            timingData.executionSeconds = std::stod(match[1].str());
        }
        std::regex regex_total("\"total_seconds\"\\s*:\\s*([0-9]*\\.?[0-9]+)");
        if(std::regex_search(line, match, regex_total)) {
            timingData.totalSeconds = std::stod(match[1].str());
        }
    }
    return timingData;
}

void writeResultsToFile(const std::string& feature, const std::string &csvFilename, bool opt, bool firstRead, const TimingData &timingData) {
    const std::string resultsFile = "evaluation/evaluation_results_" + feature + ".csv";
    bool fileExists = std::filesystem::exists(resultsFile);
    std::ofstream ofs(resultsFile, std::ios::app);
    if (!ofs) {
        throw std::runtime_error("Could not open " + resultsFile + " for writing.");
    }
    if (!fileExists) {
        ofs << "CSVFile,OptEnabled,FirstRead,NumCols,NumRows,FileType,ReadTime,WriteTime,StartupSeconds,ParsingSeconds,CompilationSeconds,ExecutionSeconds,TotalSeconds,WriteTime\n";
    }
    
    // Extract numRows, numCols, and FileType from the filename.
    // Expected format: data_<numRows>r_<numCols>c_<FileType>.csv
    std::string baseFilename = csvFilename;
    size_t pos = baseFilename.rfind(".csv");
    if (pos != std::string::npos) {
        baseFilename = baseFilename.substr(0, pos);
    }
    std::vector<std::string> parts;
    std::istringstream iss(baseFilename);
    std::string token;
    while (std::getline(iss, token, '_')) {
        parts.push_back(token);
    }
    int numRows = 0, numCols = 0;
    std::string type = "";
    if (parts.size() >= 4) {
        // parts[0] is "data", parts[1] is "<numRows>r", parts[2] is "<numCols>c", parts[3] is "<FileType>"
        std::string rowToken = parts[1]; // e.g. "1000r"
        std::string colToken = parts[2]; // e.g. "10c"
        if (!rowToken.empty() && rowToken.back() == 'r') {
            rowToken.pop_back(); // remove trailing 'r'
        }
        if (!colToken.empty() && colToken.back() == 'c') {
            colToken.pop_back(); // remove trailing 'c'
        }
        numRows = std::stoi(rowToken);
        numCols = std::stoi(colToken);
        type = parts[3];
    }
    
    std::string optStr = opt ? "true" : "false";
    std::string firstReadStr = firstRead ? "true" : "false";
    ofs << csvFilename << ","
        << optStr << ","
        << firstReadStr << ","
        << numCols << ","
        << numRows << ","
        << type << ","
        << timingData.readTime << ","
        << timingData.writeTime << ","
        << timingData.startupSeconds << ","
        << timingData.parsingSeconds << ","
        << timingData.compilationSeconds << ","
        << timingData.executionSeconds << ","
        << timingData.totalSeconds << "\n";
    ofs.close();
}

void runEvalTestCase(const std::string &csvFilename,
                     std::string feature= "posmap",
                     std::string daphneScript= "",
                     const std::string &dirPath= "evaluation/"
                     ) {
    // Remove potential binary output file.
    std::filesystem::remove(dirPath + csvFilename + "." + feature);
    if (daphneScript.empty()) {
        daphneScript = createDaphneScript(dirPath, csvFilename, csvFilename+".daphne");
    }else{
        daphneScript = dirPath + daphneScript;
    }
    
    // Normal read for comparison.
    std::string output = runDaphneEval(daphneScript, "--timing");
    std::cout << output << std::endl;
    TimingData timingData = extractTiming(output);
    writeResultsToFile(feature, csvFilename, false, true, timingData);

    // Build binary file and positional map on first read.
    output = runDaphneEval(daphneScript, "--timing", "--second-read-opt");
    std::cout << output << std::endl;
    timingData = extractTiming(output, true);
    writeResultsToFile(feature, csvFilename, true, true, timingData);
    CHECK(std::filesystem::exists(dirPath + csvFilename + "." + feature));

    // Subsequent read.
    output = runDaphneEval( daphneScript, "--timing", "--second-read-opt");
    std::cout << output << std::endl;
    timingData = extractTiming(output);
    writeResultsToFile(feature, csvFilename, true, false, timingData);
}

TEST_CASE("EvalTestCaseVariant60KB", TAG_IO) {
    // Example instantiation.
    const std::string csvFilename = "data_1000r_10c_NUMBER.csv";
    const std::string daphneScript = "evalReadFrame.daphne";
    runEvalTestCase(csvFilename, "posmap", daphneScript);
}

TEST_CASE("EvalTestCaseVariant6MB", TAG_IO) {
    // Example instantiation.
    const std::string csvFilename = "data_1000r_1000c_NUMBER.csv";
    const std::string daphneScript = "evalReadFrame2.daphne";
    runEvalTestCase(csvFilename, "posmap");//, daphneScript);
}