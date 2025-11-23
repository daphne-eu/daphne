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


# *****************************************************************************
# This section finds the required ANTLR jar file either in a provided location
# or a number of default locations and fails if no jar file is found.
# Furthermore, the default value is set here to simplify manual cmake invocations.
if(NOT DEFINED ANTLR_VERSION)
	SET(ANTLR_VERSION 4.9.2 CACHE STRING "The version of ANTLR (parsing library).")
	message(STATUS "Setting ANTLR version default (${ANTLR_VERSION}) in src/parser/CMakeLists.txt")
endif()

if(EXISTS ${ANTLR4_JAR_LOCATION})
	set(ANTLR_JAR_FOUND TRUE)
else()
	# look through provided CMAKE_PREFIX_PATHs
	foreach(path ${CMAKE_PREFIX_PATH})
		if(NOT DEFINED ANTLR_JAR_FOUND)
			set(ANTLR4_JAR_LOCATION ${path}/share/antlr4/antlr-${ANTLR_VERSION}-complete.jar CACHE STRING "default value for ANTLR jar file" FORCE)
			if(EXISTS ${ANTLR4_JAR_LOCATION})
				set(ANTLR_JAR_FOUND TRUE)
			else()
				unset(ANTLR4_JAR_LOCATION)
				unset(ANTLR_JAR_FOUND)
			endif()
		endif()
	endforeach(path)

	# fallback if not using CMAKE_PREFIX_PATH (e.g., system/container install)
	if(NOT DEFINED CMAKE_PREFIX_PATH OR NOT DEFINED ANTLR_JAR_FOUND)
		set(ANTLR4_JAR_LOCATION /usr/local/share/antlr4/antlr-${ANTLR_VERSION}-complete.jar CACHE STRING "default value for ANTLR jar file" FORCE)
		if(EXISTS ${ANTLR4_JAR_LOCATION})
			set(ANTLR_JAR_FOUND TRUE)
		else()
			unset(ANTLR4_JAR_LOCATION)
		endif()
	endif()
endif()

if(DEFINED ANTLR_JAR_FOUND)
	message(STATUS "ANTLR4_JAR_LOCATION is ${ANTLR4_JAR_LOCATION}")
else()
	message(FATAL_ERROR "ANTLR jar file not found")
endif()

# *****************************************************************************

include(GNUInstallDirs) # for ${CMAKE_INSTALL_LIBDIR}

find_package(antlr4-generator REQUIRED)
find_package(antlr4-runtime REQUIRED)

include_directories(SYSTEM ${PROJECT_BINARY_DIR} ${ANTLR4_INCLUDE_DIR})

# *****************************************************************************
# Descend to subdirectories
# *****************************************************************************

add_subdirectory(daphnedsl)
add_subdirectory(sql)
