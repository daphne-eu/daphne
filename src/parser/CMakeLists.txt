# Copyright 2021 The DAPHNE Consortium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# *****************************************************************************
# Configuration for ANTLR
# *****************************************************************************

# Information on using ANTLR's cmake packages can be found at:
# - https://github.com/antlr/antlr4/blob/master/runtime/Cpp/README.md
# - https://github.com/antlr/antlr4/blob/master/runtime/Cpp/cmake/README.md
# - https://github.com/antlr/antlr4/blob/master/runtime/Cpp/cmake/Antlr4Package.md

message(STATUS "ANTLR4_JAR_LOCATION is ${ANTLR4_JAR_LOCATION}")

include(GNUInstallDirs) # for ${CMAKE_INSTALL_LIBDIR}

find_package(antlr4-generator REQUIRED)
find_package(antlr4-runtime REQUIRED)

include_directories(${PROJECT_BINARY_DIR} ${ANTLR4_INCLUDE_DIR})

# *****************************************************************************
# Descend to subdirectories
# *****************************************************************************

add_subdirectory(daphnedsl)
add_subdirectory(sql)
