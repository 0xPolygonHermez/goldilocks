BUILD_DIR := ./build
BUILD_DIR_GPU := ./build_gpu
SRC_DIRS := ./src
TEST_DIRS := ./test

include CudaArch.mk

LIBOMP := $(shell find /usr/lib/llvm-* -name "libomp.so" | sed 's/libomp.so//')
ifndef LIBOMP
$(error LIBOMP is not set, you need to install libomp-dev)
endif

HASAVX := $(shell grep -m 1 avx /proc/cpuinfo)
HASAVX512 := $(shell grep -m 1 avx512 /proc/cpuinfo)

CXX = g++
ifeq ($(HASAVX),)
  $(info No AVX support detected)
CXXFLAGS := -std=c++17 -Wall -pthread -fopenmp
else
ifeq ($(HASAVX512),)
  $(info AVX support detected)
CXXFLAGS := -std=c++17 -Wall -pthread -fopenmp -D__USE_AVX__ -mavx2
else
  $(info AVX512 support detected)
  CXXFLAGS := -std=c++17 -Wall -pthread -fopenmp -D__USE_AVX__ -D__USE_AVX512__ -mavx2 -mavx512f
endif
endif
ARCH = $(shell uname -m)
ifeq ($(ARCH),aarch64)
 $(info ARM64 detected, using NEON by default)
CXXFLAGS += -D__USE_NEON__
endif

LDFLAGS := -lpthread -lgmp -lstdc++ -lgmpxx -lbenchmark
ASFLAGS := -felf64

CC := gcc
NVCC := /usr/local/cuda/bin/nvcc

# Debug build flags
ifeq ($(dbg),1)
      CXXFLAGS += -g
else
      CXXFLAGS += -O3
endif

### Establish the operating system name
KERNEL = $(shell uname -s)
ifneq ($(KERNEL),Linux)
 $(error "$(KERNEL), is not a valid kernel")
endif

SRCS := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.asm -or -name *.cu)
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)
ALLSRCS := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.asm -or -name *.hpp -or -name *.cu -or -name *.cuh)

INC_DIRS := $(shell find $(SRC_DIRS) -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

CPPFLAGS ?= $(INC_FLAGS) -MMD -MP

# c++ source
$(BUILD_DIR)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -c $< -o $@

# c++ source with CUDA support
$(BUILD_DIR_GPU)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) -D__USE_CUDA__ $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -c $< -o $@

$(BUILD_DIR_GPU)/%.cu.o: %.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) -D__USE_CUDA__ -DGPU_TIMING -Iutils -Xcompiler -O3 -Xcompiler -fopenmp -Xcompiler -fPIC -Xcompiler -mavx2 -arch=$(CUDA_ARCH) -dc $< --output-file $@

.PHONY: clean

testcpu: tests/tests.cpp src/goldilocks_base_field.cpp src/goldilocks_cubic_extension.cpp src/ntt_goldilocks.cpp src/poseidon_goldilocks.cpp
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -Wno-unused-but-set-variable $^ -o $@ $(LDFLAGS) -lgtest

testgpu: $(BUILD_DIR_GPU)/tests/tests.cpp.o $(BUILD_DIR)/src/goldilocks_base_field.cpp.o $(BUILD_DIR)/src/goldilocks_cubic_extension.cpp.o $(BUILD_DIR)/utils/timer_gl.cpp.o $(BUILD_DIR_GPU)/src/ntt_goldilocks.cpp.o $(BUILD_DIR)/src/poseidon_goldilocks.cpp.o $(BUILD_DIR_GPU)/src/ntt_goldilocks.cu.o $(BUILD_DIR_GPU)/src/poseidon_goldilocks.cu.o $(BUILD_DIR_GPU)/utils/cuda_utils.cu.o
	$(NVCC) -Xcompiler -O3 -Xcompiler -fopenmp -arch=$(CUDA_ARCH) -o $@ $^ -lgtest -lgmp

runtestcpu: testcpu
	./testcpu --gtest_filter=GOLDILOCKS_TEST.merkletree_seq

runtestgpu: testgpu
	./testgpu --gtest_filter=GOLDILOCKS_TEST.merkletree_cuda

full: $(BUILD_DIR_GPU)/tests/tests.cu.o $(BUILD_DIR_GPU)/src/goldilocks_base_field.cpp.o  $(BUILD_DIR_GPU)/utils/timer_gl.cpp.o $(BUILD_DIR_GPU)/utils/cuda_utils.cu.o  $(BUILD_DIR_GPU)/src/ntt_goldilocks.cpp.o $(BUILD_DIR_GPU)/src/poseidon_goldilocks.cpp.o $(BUILD_DIR_GPU)/src/ntt_goldilocks.cu.o $(BUILD_DIR_GPU)/src/poseidon_goldilocks.cu.o
	$(NVCC) -Xcompiler -O3 -Xcompiler -fopenmp -arch=$(CUDA_ARCH) -o $@ $^ -lgtest -lgmp

runfullgpu: full
	./full --gtest_filter=GOLDILOCKS_TEST.full_gpu

runfullcpu: full
	./full --gtest_filter=GOLDILOCKS_TEST.full_cpu

benchcpu: benchs/bench.cpp src/goldilocks_base_field.cpp src/goldilocks_cubic_extension.cpp src/ntt_goldilocks.cpp src/poseidon_goldilocks.cpp
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

benchgpu: $(BUILD_DIR_GPU)/benchs/bench.cpp.o $(BUILD_DIR)/src/goldilocks_base_field.cpp.o $(BUILD_DIR)/src/goldilocks_cubic_extension.cpp.o $(BUILD_DIR_GPU)/src/poseidon_goldilocks.cpp.o $(BUILD_DIR_GPU)/src/ntt_goldilocks.cu.o $(BUILD_DIR_GPU)/src/poseidon_goldilocks.cu.o
	$(NVCC) -Xcompiler -O3 -Xcompiler -fopenmp -arch=$(CUDA_ARCH) -o $@ $^ -lgtest -lgmp -lbenchmark

runbenchcpu: benchcpu
	./benchcpu --benchmark_filter=MERKLETREE_BENCH_AVX

runbenchgpu: benchgpu
	./benchgpu --benchmark_filter=MERKLETREE_BENCH_CUDA

clean:
	$(RM) -r $(BUILD_DIR)
	$(RM) -r $(BUILD_DIR_GPU)
	$(RM) test
	$(RM) bench
	$(RM) testcpu
	$(RM) testgpu
	$(RM) benchcpu
	$(RM) benchgpu


-include $(DEPS)

MKDIR_P ?= mkdir -p
