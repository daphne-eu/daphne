# Copyright 2024 The DAPHNE Consortium
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

CXX = g++
CXXFLAGS = -DLIKWID_PERFMON -ggdb3 -fPIC -fno-omit-frame-pointer -O3 -march=native -fopenmp -std=c++17
ASMFLAGS = -fverbose-asm -S

INCLUDES = -I/usr/local/include/ -I../../../../../

SRCS = gemv.cpp
OBJS = $(SRCS:.cpp=.o)
TARGET = gemv.so

.PHONY: clean

all: $(TARGET)
	@echo " ==> Built target $(TARGET)"

$(TARGET): $(OBJS)
	@echo " ==> COMPILING  $@"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -shared -o $(TARGET) $(OBJS) -llikwid
	
asm: $(SRCS)
	@echo " ==> COMPILING  $@"
	$(CXX) $(ASMFLAGS) $(CXXFLAGS) $(INCLUDES) -o $(SRCS:.cpp=.s) -cpp $< -llikwid

.cpp.o:
	@echo " ==> COMPILING  $@"
	$(CXX) -c $(CXXFLAGS) $(INCLUDES) -o $@ -cpp $< -llikwid

clean:
	@echo "==> CLEANING"
	$(RM) *.o *.s *.so
