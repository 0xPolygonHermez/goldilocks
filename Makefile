# Directories
BUILD_DIR := ./build
SRC_DIR := ./src
TEST_DIR := ./tests

# find libomp
LIBOMP := $(shell find /usr/lib/llvm-* -name "libomp.so" | sed 's/libomp.so//')
ifndef LIBOMP
$(error LIBOMP is not set, you need to install libomp-dev)
endif

# Compiler and flags
CXX = g++
CXXFLAGS := -std=c++17 -Wall -pthread -fopenmp -mavx2
LDFLAGS := -lpthread -lgmp -lstdc++ -lgmpxx -lbenchmark -lgtest -lgomp  
ASFLAGS := -felf64
NVCC := /usr/local/cuda/bin/nvcc

# Preprocessor flags
CXXFLAGS += -D__AVX2__
#CXXFLAGS += -D__USE_ASSEMBLY__
#CXXFLAGS += -D__AVX512__

# Debug build flags
ifeq ($(dbg),1)
      CXXFLAGS += -g
else
      CXXFLAGS += -O3
endif

# Source files
SRCS := $(shell find $(SRC_DIR) -name *.cpp)
OBJS := $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRCS))
DEPS := $(OBJS:.o=.d)

# Include directories
INC_DIRS := $(shell find $(SRC_DIR) -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

# Preprocessor flags
CPPFLAGS := $(INC_FLAGS) -MMD -MP

# Targets
all: testscpu

# Linking the test executable
testscpu: $(OBJS) $(BUILD_DIR)/tests/tests.o
	$(CXX) $(OBJS) $(BUILD_DIR)/tests/tests.o $(LDFLAGS) -o $@

# Compiling source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

# Compiling test files
$(BUILD_DIR)/tests/%.o: $(TEST_DIR)/%.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

# Create directories
MKDIR_P := mkdir -p

.PHONY: clean


testsgpu: $(BUILD_DIR_GPU)/tests/tests.cpp.o $(BUILD_DIR)/src/goldilocks_base_field.cpp.o $(BUILD_DIR)/src/goldilocks_cubic_extension.cpp.o $(BUILD_DIR)/utils/timer_gl.cpp.o $(BUILD_DIR_GPU)/src/ntt_goldilocks.cpp.o $(BUILD_DIR)/src/poseidon_goldilocks.cpp.o $(BUILD_DIR_GPU)/src/ntt_goldilocks.cu.o $(BUILD_DIR_GPU)/src/poseidon_goldilocks.cu.o $(BUILD_DIR_GPU)/utils/cuda_utils.cu.o
	$(NVCC) -Xcompiler -O3 -Xcompiler -fopenmp -arch=$(CUDA_ARCH) -o $@ $^ -lgtest -lgmp

benchscpu: benchs/bench.cpp $(ALLSRCS)
	$(CXX) benchs/bench.cpp src/*.cpp -lbenchmark -lpthread -lgmp  -std=c++17 -Wall -pthread -fopenmp -mavx2 -O3 -o $@


clean:
	$(RM) -r $(BUILD_DIR)
	$(RM) -r $(BUILD_DIR_GPU)
	$(RM) testscpu
	$(RM) benchscpu

-include $(DEPS)

MKDIR_P ?= mkdir -p
