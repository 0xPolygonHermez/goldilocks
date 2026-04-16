.PHONY: all clean run run-sudo run2 run3 run4

CXX        = clang++
CXXFLAGS   = -std=c++17 -O0 -fobjc-arc -ObjC++
MTL_FRAMEWORKS = -framework Metal -framework Foundation

# Goldilocks library headers (for oracle comparison in probes 2-4)
GL_SRC     = ../../src
GMP_PREFIX = $(shell brew --prefix gmp 2>/dev/null || echo /usr/local)
GL_INC     = -I$(GL_SRC) -I$(GL_SRC)/simd -I$(GMP_PREFIX)/include
GL_LIBS    = -L$(GMP_PREFIX)/lib -lgmp -lgmpxx
# Metal kernel source root (runtime newLibraryWithSource compilation)
MTL_KERNEL_DIR = $(GL_SRC)/metal/kernels

# OpenMP: needed because poseidon_goldilocks.cpp uses omp_get_max_threads.
OMP_FLAGS  = -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include \
             -L/opt/homebrew/opt/libomp/lib -lomp

all: probe_device probe_field probe_poseidon probe_merkle bench_merkle bench_ntt

probe_device: probe_device.mm
	$(CXX) $(CXXFLAGS) $(MTL_FRAMEWORKS) $< -o $@

# Goldilocks runtime objects (static constants + Poseidon seq implementations)
GL_OBJ     = $(GL_SRC)/goldilocks_base_field.cpp $(GL_SRC)/poseidon_goldilocks.cpp

# Probe 2: Goldilocks field arithmetic bit-exact verification
# Kernel source compiled at runtime via newLibraryWithSource:options:error:
# -DGOLDILOCKS_NO_METAL suppresses auto-enable of metal in poseidon_goldilocks.cpp
# (the probe uses CPU oracle only; it dispatches field.metal via runtime API).
probe_field: probe_field.mm
	$(CXX) $(CXXFLAGS) $(MTL_FRAMEWORKS) $(GL_INC) $(OMP_FLAGS) \
	    -DGOLDILOCKS_NO_METAL \
	    -DMTL_KERNEL_DIR=\"$(MTL_KERNEL_DIR)\" \
	    $< $(GL_OBJ) $(GL_LIBS) -o $@

# Probe 3: single-state pod12 Poseidon bit-exact verification
# -DGOLDILOCKS_NO_METAL: same reason as probe_field.
probe_poseidon: probe_poseidon.mm
	$(CXX) $(CXXFLAGS) $(MTL_FRAMEWORKS) $(GL_INC) $(OMP_FLAGS) \
	    -DGOLDILOCKS_NO_METAL \
	    -DMTL_KERNEL_DIR=\"$(MTL_KERNEL_DIR)\" \
	    $< $(GL_OBJ) $(GL_LIBS) -o $@

# Probe 4: end-to-end Merkle oracle (128 cols x 64 rows Fibonacci input).
# Requires a pre-built goldilocks.metallib in the current directory.
# Build the metallib with:
#   make goldilocks.metallib
# or manually:
#   xcrun metal -c $(MTL_KERNEL_DIR)/field.metal    -o field.air
#   xcrun metal -c $(MTL_KERNEL_DIR)/poseidon.metal -o poseidon.air
#   xcrun metallib field.air poseidon.air -o goldilocks.metallib
METAL_BRIDGE_SRCS = $(GL_SRC)/metal/metal_context.mm \
                    $(GL_SRC)/metal/poseidon_metal.mm

# goldilocks.metallib is a runtime dependency (probe loads it at start-up)
# but NOT a compile-time dependency — the binary compiles without it.
# Build it separately with: make goldilocks.metallib
probe_merkle: probe_merkle.mm $(METAL_BRIDGE_SRCS)
	$(CXX) $(CXXFLAGS) $(MTL_FRAMEWORKS) \
	    $(GL_INC) $(OMP_FLAGS) \
	    -DGOLDILOCKS_HAS_METAL \
	    -DMTL_KERNEL_DIR=\"$(MTL_KERNEL_DIR)\" \
	    probe_merkle.mm $(METAL_BRIDGE_SRCS) $(GL_OBJ) \
	    $(GL_LIBS) -o $@

# Throughput comparison: seq / NEON / Metal on merkletree_metal-sized inputs.
bench_merkle: bench_merkle.mm $(METAL_BRIDGE_SRCS)
	$(CXX) -std=c++17 -O3 -fobjc-arc -ObjC++ $(MTL_FRAMEWORKS) \
	    $(GL_INC) $(OMP_FLAGS) \
	    -DGOLDILOCKS_HAS_METAL \
	    -DMTL_KERNEL_DIR=\"$(MTL_KERNEL_DIR)\" \
	    bench_merkle.mm $(METAL_BRIDGE_SRCS) $(GL_OBJ) \
	    $(GL_LIBS) -o $@

# Batched Merkle throughput: sequential vs batched.
bench_merkle_batch: bench_merkle_batch.mm $(METAL_BRIDGE_SRCS)
	$(CXX) -std=c++17 -O3 -fobjc-arc -ObjC++ $(MTL_FRAMEWORKS) \
	    $(GL_INC) $(OMP_FLAGS) \
	    -DGOLDILOCKS_HAS_METAL \
	    -DMTL_KERNEL_DIR=\"$(MTL_KERNEL_DIR)\" \
	    bench_merkle_batch.mm $(METAL_BRIDGE_SRCS) $(GL_OBJ) \
	    $(GL_LIBS) -o $@

# Throughput comparison: CPU (NEON fast path) vs Metal on forward NTT.
# Links ntt_metal.mm + metal_context.mm + poseidon_metal.mm (bridge sources)
# plus NTT CPU core (ntt_goldilocks.cpp) and field/Poseidon deps.
NTT_CPU_OBJ = $(GL_OBJ) $(GL_SRC)/ntt_goldilocks.cpp
bench_ntt: bench_ntt.mm $(METAL_BRIDGE_SRCS) $(GL_SRC)/metal/ntt_metal.mm
	$(CXX) -std=c++17 -O3 -fobjc-arc -ObjC++ $(MTL_FRAMEWORKS) \
	    $(GL_INC) $(OMP_FLAGS) \
	    -DGOLDILOCKS_HAS_METAL \
	    -DMTL_KERNEL_DIR=\"$(MTL_KERNEL_DIR)\" \
	    bench_ntt.mm $(METAL_BRIDGE_SRCS) $(GL_SRC)/metal/ntt_metal.mm \
	    $(NTT_CPU_OBJ) $(GL_LIBS) -o $@

# LDE (extendPol) correctness + perf: CPU vs Metal on Fibonacci-seeded input.
bench_lde: bench_lde.mm $(METAL_BRIDGE_SRCS) $(GL_SRC)/metal/ntt_metal.mm
	$(CXX) -std=c++17 -O3 -fobjc-arc -ObjC++ $(MTL_FRAMEWORKS) \
	    $(GL_INC) $(OMP_FLAGS) \
	    -DGOLDILOCKS_HAS_METAL \
	    -DMTL_KERNEL_DIR=\"$(MTL_KERNEL_DIR)\" \
	    bench_lde.mm $(METAL_BRIDGE_SRCS) $(GL_SRC)/metal/ntt_metal.mm \
	    $(NTT_CPU_OBJ) $(GL_LIBS) -o $@

# Unified-memory contention probe: run CPU NEON + GPU Metal concurrently on
# independent inputs of the same shape, compare to serial sum. Answers
# whether a hybrid dispatch would help or just fight over memory bandwidth.
bench_contention_merkle: bench_contention_merkle.mm $(METAL_BRIDGE_SRCS)
	$(CXX) -std=c++17 -O3 -fobjc-arc -ObjC++ $(MTL_FRAMEWORKS) \
	    $(GL_INC) $(OMP_FLAGS) \
	    -DGOLDILOCKS_HAS_METAL \
	    -DMTL_KERNEL_DIR=\"$(MTL_KERNEL_DIR)\" \
	    bench_contention_merkle.mm $(METAL_BRIDGE_SRCS) \
	    $(GL_OBJ) $(GL_LIBS) -o $@

bench_contention_ntt: bench_contention_ntt.mm $(METAL_BRIDGE_SRCS) $(GL_SRC)/metal/ntt_metal.mm
	$(CXX) -std=c++17 -O3 -fobjc-arc -ObjC++ $(MTL_FRAMEWORKS) \
	    $(GL_INC) $(OMP_FLAGS) \
	    -DGOLDILOCKS_HAS_METAL \
	    -DMTL_KERNEL_DIR=\"$(MTL_KERNEL_DIR)\" \
	    bench_contention_ntt.mm $(METAL_BRIDGE_SRCS) $(GL_SRC)/metal/ntt_metal.mm \
	    $(NTT_CPU_OBJ) $(GL_LIBS) -o $@

# Leaf-kernel-only timing to separate GPU execution from host overhead.
probe_kernel_gpu_time: probe_kernel_gpu_time.mm $(METAL_BRIDGE_SRCS)
	$(CXX) -std=c++17 -O3 -fobjc-arc -ObjC++ $(MTL_FRAMEWORKS) \
	    $(GL_INC) \
	    -DGOLDILOCKS_HAS_METAL \
	    -DMTL_KERNEL_DIR=\"$(MTL_KERNEL_DIR)\" \
	    probe_kernel_gpu_time.mm $(METAL_BRIDGE_SRCS) \
	    $(GL_SRC)/goldilocks_base_field.cpp $(GL_LIBS) -o $@

# Deep profiling: PSO introspection + GPU-kernel-only timing at varied tpg.
probe_profile: probe_profile.mm
	$(CXX) -std=c++17 -O3 -fobjc-arc -ObjC++ $(MTL_FRAMEWORKS) \
	    $(GL_INC) \
	    -DGOLDILOCKS_HAS_METAL \
	    -DMTL_KERNEL_DIR=\"$(MTL_KERNEL_DIR)\" \
	    probe_profile.mm \
	    $(GL_SRC)/goldilocks_base_field.cpp $(GL_LIBS) -o $@

# Build the .metallib from Part 4 kernel sources.
# constants.metal.inc must exist in $(MTL_KERNEL_DIR); it is generated by Part 3.
goldilocks.metallib: $(MTL_KERNEL_DIR)/field.metal \
                     $(MTL_KERNEL_DIR)/poseidon.metal
	xcrun metal -c $(MTL_KERNEL_DIR)/field.metal    -o field.air
	xcrun metal -c $(MTL_KERNEL_DIR)/poseidon.metal \
	    -I$(MTL_KERNEL_DIR) -o poseidon.air
	xcrun metallib field.air poseidon.air -o goldilocks.metallib

run: all run2 run3 run4
	@echo "=== probe 1: Metal device ==="
	-./probe_device ; echo "exit=$$?"

run-sudo: all
	@echo "=== probe 1 (sudo): Metal device ==="
	-sudo ./probe_device ; echo "exit=$$?"

run2: probe_field
	@echo "=== probe 2: field arithmetic ==="
	-./probe_field ; echo "exit=$$?"

run3: probe_poseidon
	@echo "=== probe 3: pod12 Poseidon ==="
	-./probe_poseidon ; echo "exit=$$?"

run4: probe_merkle
	@echo "=== probe 4: Merkle oracle ==="
	-./probe_merkle ; echo "exit=$$?"

clean:
	rm -f probe_device probe_field probe_poseidon probe_merkle \
	      *.air goldilocks.metallib
