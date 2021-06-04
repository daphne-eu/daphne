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

#include <api/cli/StatusCode.h>

#include <catch.hpp>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

std::string readTextFile(const std::string & filePath) {
    std::ifstream ifs(filePath, std::ios::in);
    if (!ifs.good())
        throw std::runtime_error("could not open file '" + filePath + "'");
    
    std::stringstream stream;
    stream << ifs.rdbuf();
    
    return stream.str();
}

template<typename... Args>
int runProgram(std::stringstream & out, std::stringstream & err, const char * execPath, Args ... args) {
    int linkOut[2]; // pipe ends for stdout
    int linkErr[2]; // pipe ends for stderr
    char buf[1024]; // internal buffer for reading from the pipes
    
    // Try to create the pipes.
    if(pipe(linkOut) == -1)
        throw std::runtime_error("could not create pipe");
    if(pipe(linkErr) == -1)
        throw std::runtime_error("could not create pipe");
    
    // Try to create the child process.
    pid_t p = fork();
    
    if(p == -1)
        throw std::runtime_error("could not create child process");
    else if(p) { // parent
        // Close write end of pipes.
        close(linkOut[1]);
        close(linkErr[1]);
        
        // Read data from stdout and stderr of the child from the pipes.
        ssize_t numBytes;
        while(numBytes = read(linkOut[0], buf, sizeof(buf)))
            out.write(buf, numBytes);
        while(numBytes = read(linkErr[0], buf, sizeof(buf)))
            err.write(buf, numBytes);
        
        // Wait for child's termination.
        int status;
        waitpid(p, &status, 0);
        return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    }
    else { // child
        // Redirect stdout and stderr to the pipe.
        dup2(linkOut[1], STDOUT_FILENO);
        dup2(linkErr[1], STDERR_FILENO);
        close(linkOut[0]);
        close(linkOut[1]);
        close(linkErr[0]);
        close(linkErr[1]);
        
        // Execute other program.
        execl(execPath, args..., static_cast<char *>(nullptr));
        
        // execl does not return, unless it failed.
        throw std::runtime_error("could not execute the program");
    }
}

int runDaphne(std::stringstream & out, std::stringstream & err, const char * scriptPath) {
    return runProgram(out, err, "build/bin/daphnec", "daphnec", scriptPath);
}

void checkDaphneStatusCode(const std::string & scriptFilePath, StatusCode exp) {
    std::stringstream out;
    std::stringstream err;
    int status = runDaphne(out, err, scriptFilePath.c_str());

    REQUIRE(status == exp);
}

void checkDaphneStatusCode(const std::string & dirPath, const std::string & name, unsigned idx, StatusCode exp) {
    checkDaphneStatusCode(dirPath + name + '_' + std::to_string(idx) + ".daphne", exp);
}

void compareDaphneToRef(const std::string & scriptFilePath, const std::string & refFilePath) {
    std::stringstream out;
    std::stringstream err;
    int status = runDaphne(out, err, scriptFilePath.c_str());

    REQUIRE(status == StatusCode::SUCCESS);
    CHECK(out.str() == readTextFile(refFilePath.c_str()));
    CHECK(err.str().empty());
}

void compareDaphneToRef(const std::string & dirPath, const std::string & name, unsigned idx) {
    const std::string filePath = dirPath + name + '_' + std::to_string(idx);
    compareDaphneToRef(filePath + ".daphne", filePath + ".txt");
}