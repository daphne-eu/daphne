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
#include <runtime/distributed/worker/WorkerImpl.h>

#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>

#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

std::string readTextFile(const std::string &filePath) {
    std::ifstream ifs(filePath, std::ios::in);
    if (!ifs.good())
        throw std::runtime_error("could not open file '" + filePath + "'");

    std::stringstream stream;
    stream << ifs.rdbuf();

    return stream.str();
}

std::string generalizeDataTypes(const std::string &str) {
    std::regex re("(DenseMatrix|CSRMatrix)");
    return std::regex_replace(str, re, "<SomeMatrix>");
}

bool compareFileContents(const std::string &filePath1, const std::string &filePath2) {
    std::ifstream file1(filePath1, std::ios::binary);
    std::ifstream file2(filePath2, std::ios::binary);
    if (!file1.is_open() || !file2.is_open()) {
        std::cerr << "Cannot open one or both files." << std::endl;
        return false;
    }
    std::string line1, line2;
    bool filesAreEqual = true;
    while (std::getline(file1, line1)) {
        if (!std::getline(file2, line2)) {
            filesAreEqual = false;
            break;
        }
        if (line1 != line2) {
            filesAreEqual = false;
            break;
        }
    }
    if (filesAreEqual && std::getline(file2, line2)) {
        if (!line2.empty()) {
            filesAreEqual = false;
        }
    }
    file1.close();
    file2.close();
    return filesAreEqual;
}