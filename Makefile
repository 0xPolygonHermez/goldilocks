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
CXXFLAGS := -std=c++17 -Wall -pthread -fopenmp #-mavx2
LDFLAGS := -lpthread -lgmp -lstdc++ -lgmpxx -lbenchmark -lgtest -lgomp  

# Preprocessor flags
CXXFLAGS += -D__AVX2__
CXXFLAGS += -D__USE_ASSEMBLY__
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

# Linking the benchmark executable
benchscpu: $(BUILD_DIR)/benchs/bench.o $(OBJS)
	$(CXX) $(BUILD_DIR)/benchs/bench.o $(OBJS) $(CXXFLAGS) $(LDFLAGS) -o $@

# Compiling source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

# Compiling benchmark files
$(BUILD_DIR)/benchs/%.o: benchs/%.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

# Compiling test files
$(BUILD_DIR)/tests/%.o: $(TEST_DIR)/%.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

.PHONY: clean

clean:
	$(RM) -r $(BUILD_DIR)
	$(RM) testscpu
	$(RM) benchscpu

-include $(DEPS)

MKDIR_P ?= mkdir -p
