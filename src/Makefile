# Makefile for compiling naive_direct_conv2d

# Variables
MACHINE_NAME := $(shell uname -m)
CXX := g++
CXXFLAGS := -march=native -lcpuinfo -lclog -fconcepts-ts -std=c++17 -Ofast
INCLUDES := -I/usr/include/$(MACHINE_NAME)-linux-gnu/
LIBS := -L/usr/include/$(MACHINE_NAME)-linux-gnu -lopenblas

# Executable names
EXES := hipack_st.exe hipack_mt.exe
# Source files
SRCS := hipack.cpp

# Default target
all: $(EXES)

# Compile and link each executable
hipack_st.exe: hipack.cpp
	$(CXX) -o $@ $< $(CXXFLAGS) $(INCLUDES) $(LIBS)

hipack_mt.exe: hipack.cpp
	$(CXX) -DENABLE_OPENMP -fopenmp -lgomp -o $@ $< $(CXXFLAGS) $(INCLUDES) $(LIBS)



# Clean target
clean:
	rm -f $(EXES)

# Phony targets
.PHONY: all clean