TARGET_EXEC := example

BUILD_DIR := ./build
BUILD_DIR_GPU := ./build_gpu
SRC_DIRS := ./src
TEST_DIRS := ./test

-include CudaArch.mk

### Detect OS and architecture
KERNEL := $(shell uname -s)
ARCH   := $(shell uname -m)

### Detect Homebrew prefix (macOS only) and locate libomp
ifeq ($(KERNEL),Darwin)
  HOMEBREW_PREFIX := $(shell brew --prefix 2>/dev/null)
  LIBOMP_PREFIX   := $(shell brew --prefix libomp 2>/dev/null)
  LIBOMP_INC      := $(LIBOMP_PREFIX)/include
  LIBOMP_LIB      := $(LIBOMP_PREFIX)/lib
  LIBOMP_FLAGS    := -Xpreprocessor -fopenmp -I$(LIBOMP_INC) -L$(LIBOMP_LIB) -lomp
else
  LIBOMP := $(shell find /usr/lib/llvm-* -name "libomp.so" 2>/dev/null | sed 's/libomp.so//')
  ifndef LIBOMP
    $(error LIBOMP is not set, you need to install libomp-dev)
  endif
  LIBOMP_FLAGS := -L$(LIBOMP) -fopenmp
endif

### GMP and GTest locations
ifeq ($(KERNEL),Darwin)
  GMP_PREFIX   := $(shell brew --prefix gmp 2>/dev/null)
  GTEST_PREFIX := $(shell brew --prefix googletest 2>/dev/null)
  BENCH_PREFIX := $(shell brew --prefix google-benchmark 2>/dev/null)
  GMP_FLAGS    := -I$(GMP_PREFIX)/include -L$(GMP_PREFIX)/lib
  GTEST_FLAGS  := -I$(GTEST_PREFIX)/include -L$(GTEST_PREFIX)/lib
  BENCH_FLAGS  := -I$(BENCH_PREFIX)/include -L$(BENCH_PREFIX)/lib
else
  GMP_FLAGS    :=
  GTEST_FLAGS  :=
  BENCH_FLAGS  :=
endif

### Architecture-specific SIMD flags
ifeq ($(ARCH),x86_64)
  SIMD_FLAGS := -mavx2
else
  SIMD_FLAGS :=
endif

### Compiler selection
ifeq ($(KERNEL),Darwin)
  CXX := clang++
  CC  := clang
else
  CXX := g++
  CC  := gcc
endif

CXXFLAGS := -std=c++17 -Wall -pthread $(LIBOMP_FLAGS)
LDFLAGS  := -lpthread $(GMP_FLAGS) -lgmp -lstdc++ -lgmpxx -lbenchmark
ASFLAGS := -felf64

NVCC := /usr/local/cuda/bin/nvcc

# Debug build flags
ifeq ($(dbg),1)
      CXXFLAGS += -g
else
      CXXFLAGS += -O3
endif

SRCS := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.asm -or -name *.cu)
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)
ALLSRCS := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.asm -or -name *.hpp -or -name *.cu -or -name *.cuh)

INC_DIRS := $(shell find $(SRC_DIRS) -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

CPPFLAGS ?= $(INC_FLAGS) -MMD -MP $(SIMD_FLAGS)

testcpu: tests/tests.cpp $(ALLSRCS)
	$(CXX) -std=c++17 tests/tests.cpp src/*.cpp $(GTEST_FLAGS) $(GMP_FLAGS) $(LIBOMP_FLAGS) -lgtest -lgmp -O3 -Wall -pthread $(SIMD_FLAGS) -o $@

$(BUILD_DIR)/$(TARGET_EXEC): $(OBJS)
	$(CXX) $(OBJS) $(CXXFLAGS) -o $@ $(LDFLAGS)

# c++ source
$(BUILD_DIR)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -c $< -o $@

# c++ source with CUDA support
$(BUILD_DIR_GPU)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) -D__USE_CUDA__ $(SIMD_FLAGS) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -c $< -o $@

$(BUILD_DIR_GPU)/%.cu.o: %.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) -D__USE_CUDA__ -DGPU_TIMING -Iutils -Xcompiler -O3 -Xcompiler -fopenmp -Xcompiler -fPIC -Xcompiler $(SIMD_FLAGS) -arch=$(CUDA_ARCH) -dc $< --output-file $@

.PHONY: clean


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

benchcpu: benchs/bench.cpp src/*.cpp
	$(CXX) $^ $(GMP_FLAGS) $(GTEST_FLAGS) $(BENCH_FLAGS) $(LIBOMP_FLAGS) -lbenchmark -lpthread -lgmp -std=c++17 -Wall -pthread $(SIMD_FLAGS) -O3 -o $@

benchgpu: $(BUILD_DIR_GPU)/benchs/bench.cpp.o $(BUILD_DIR)/src/goldilocks_base_field.cpp.o $(BUILD_DIR)/src/goldilocks_cubic_extension.cpp.o $(BUILD_DIR_GPU)/src/poseidon_goldilocks.cpp.o $(BUILD_DIR_GPU)/src/ntt_goldilocks.cu.o $(BUILD_DIR_GPU)/src/poseidon_goldilocks.cu.o
	$(NVCC) -Xcompiler -O3 -Xcompiler -fopenmp -arch=$(CUDA_ARCH) -o $@ $^ -lgtest -lgmp -lbenchmark

runbenchcpu: benchcpu
	./benchcpu --benchmark_filter=MERKLETREE_BENCH_AVX

runbenchcpu_neon: benchcpu
	./benchcpu --benchmark_filter='_NEON'

runbenchgpu: benchgpu
	./benchgpu --benchmark_filter=MERKLETREE_BENCH_CUDA

# ============================================================================
# Metal (Darwin + GOLDILOCKS_HAS_METAL) — Apple Silicon only
# ============================================================================
ifeq ($(KERNEL),Darwin)

SRCS_CPP := $(filter-out ./src/tests/%,$(filter %.cpp,$(SRCS)))

METAL_FLAGS := -framework Metal -framework Foundation -framework QuartzCore
METAL_CXX   := $(CXX) -fobjc-arc -std=c++17 -DGOLDILOCKS_HAS_METAL
METAL_SHADERS_DIR := src/metal/kernels
METAL_SHADERS := $(METAL_SHADERS_DIR)/field.metal $(METAL_SHADERS_DIR)/poseidon.metal $(METAL_SHADERS_DIR)/ntt.metal
METAL_AIR := $(patsubst %.metal,$(BUILD_DIR)/%.air,$(METAL_SHADERS))
METALLIB := $(BUILD_DIR)/goldilocks.metallib
METAL_CONSTS := $(METAL_SHADERS_DIR)/constants.metal.inc
METAL_MM_SRCS := src/metal/metal_context.mm src/metal/poseidon_metal.mm src/metal/ntt_metal.mm
METAL_MM_OBJS := $(patsubst %.mm,$(BUILD_DIR)/%.mm.o,$(METAL_MM_SRCS))

# Codegen for constants
$(METAL_CONSTS): tools/gen_metal_constants.cpp src/poseidon_goldilocks_constants.hpp
	@mkdir -p $(BUILD_DIR) $(METAL_SHADERS_DIR)
	$(CXX) -std=c++17 -O0 -Isrc $(GMP_FLAGS) tools/gen_metal_constants.cpp -o $(BUILD_DIR)/gen_metal_constants
	$(BUILD_DIR)/gen_metal_constants $(METAL_CONSTS)

# .metal -> .air (needs constants.metal.inc)
$(BUILD_DIR)/%.air: %.metal $(METAL_CONSTS)
	@mkdir -p $(dir $@)
	xcrun -sdk macosx metal -std=metal3.0 -c -I$(METAL_SHADERS_DIR) $< -o $@

# .air -> .metallib
$(METALLIB): $(METAL_AIR)
	xcrun -sdk macosx metallib $^ -o $@

# .mm -> .o
$(BUILD_DIR)/%.mm.o: %.mm
	@mkdir -p $(dir $@)
	$(METAL_CXX) -I. -Isrc $(GMP_FLAGS) $(LIBOMP_FLAGS) -c $< -o $@

# testmetal target
testmetal: $(METALLIB) $(METAL_MM_OBJS) src/tests/tests_metal.cpp $(SRCS_CPP)
	$(METAL_CXX) -O3 -I. -Isrc $(GTEST_FLAGS) $(GMP_FLAGS) $(LIBOMP_FLAGS) \
	  src/tests/tests_metal.cpp $(SRCS_CPP) $(METAL_MM_OBJS) \
	  -o $@ -lgtest -lgmp -lstdc++ -lpthread $(METAL_FLAGS)

runtestmetal: testmetal $(METALLIB)
	cp $(METALLIB) goldilocks.metallib
	./testmetal --gtest_filter=GOLDILOCKS_TEST.merkletree_metal

clean-metal:
	rm -f $(METAL_AIR) $(METALLIB) $(METAL_MM_OBJS) $(METAL_CONSTS) testmetal goldilocks.metallib

# Hook into top-level clean so `make clean` removes Metal artifacts too.
clean: clean-metal

.PHONY: testmetal runtestmetal clean-metal

endif  # Darwin
# ============================================================================

# Standalone NEON mul fuzz harness (Phase 1 Part 2 verification)
check_neon_mul: tests/check_neon_mul.cpp src/goldilocks_base_field.cpp
	$(CXX) -std=c++17 $^ $(GMP_FLAGS) -lgmp -O2 -o $@

runcheck_neon_mul: check_neon_mul
	./check_neon_mul

# Throughput microbenchmark for NEON mul
check_neon_mul_bench: benchs/check_neon_mul_bench.cpp src/goldilocks_base_field.cpp
	$(CXX) -std=c++17 $^ $(GMP_FLAGS) -lgmp -O3 -o $@

runcheck_neon_mul_bench: check_neon_mul_bench
	./check_neon_mul_bench

# SIMD traits smoke test (Part 1 verification)
check_simd_traits_compile: tests/check_simd_traits_compile.cpp src/goldilocks_base_field.cpp
	$(CXX) -std=c++17 $^ $(GMP_FLAGS) -lgmp -O2 -o $@

runcheck_simd_traits: check_simd_traits_compile
	./check_simd_traits_compile

clean:
	$(RM) -r $(BUILD_DIR)
	$(RM) -r $(BUILD_DIR_GPU)
	$(RM) test
	$(RM) bench
	$(RM) testcpu
	$(RM) testgpu
	$(RM) benchcpu
	$(RM) benchgpu
	$(RM) check_neon_mul
	$(RM) check_neon_mul_bench
	$(RM) check_simd_traits_compile


-include $(DEPS)

MKDIR_P ?= mkdir -p
