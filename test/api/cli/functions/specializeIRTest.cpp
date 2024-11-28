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

#include <tags.h>

#include <fstream>

#include <catch.hpp>

#include <string>

const std::string dirPath = "test/api/cli/functions/";

#define MAKE_TEST_CASE(name)                                                                    \
    TEST_CASE(name, TAG_FUNCTIONS) {                                                            \
	std::stringstream err;									\
	std::stringstream out;                                                                  \
	DYNAMIC_SECTION(name << ".daphne") { runDaphne(&out,&err, name, "--explain property_inference"); }			\
	std::string expected = "";								\
	std::string line;									\
	std::ifstream file(name+".txt");							\
	while(std::getline(file,line)) {							\
		expected += line;								\
	}											\
	file.close();										\
	REQUIRE(err+out == expected);            						\
    }                                                                     	                \
}

MAKE_TEST_CASE("specializeIRTest")
