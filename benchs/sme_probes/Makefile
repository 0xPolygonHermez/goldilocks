.PHONY: all clean run run-sudo

CXX      = clang++
CXXFLAGS = -std=c++17 -O0
SME_ARCH = -march=armv9-a+sme
SME_I64_ARCH = -march=armv9-a+sme+sme-i16i64 -mcpu=apple-m4

all: 01_stream_nop 02_zero_za 03_i16_i64_matmul

01_stream_nop: 01_stream_nop.cpp
	$(CXX) $(CXXFLAGS) $(SME_ARCH) $< -o $@

02_zero_za: 02_zero_za.cpp
	$(CXX) $(CXXFLAGS) $(SME_ARCH) $< -o $@

03_i16_i64_matmul: 03_i16_i64_matmul.cpp
	$(CXX) -std=c++17 -O2 $(SME_I64_ARCH) $< -o $@

run: all
	@echo "=== probe 1: streaming NOP ==="
	-./01_stream_nop ; echo "exit=$$?"
	@echo
	@echo "=== probe 2: zero ZA ==="
	-./02_zero_za ; echo "exit=$$?"
	@echo
	@echo "=== probe 3: SME i16×i16→i64 matmul ==="
	-./03_i16_i64_matmul ; echo "exit=$$?"

run-sudo: all
	@echo "=== probe 1 (sudo): streaming NOP ==="
	-sudo ./01_stream_nop ; echo "exit=$$?"
	@echo
	@echo "=== probe 2 (sudo): zero ZA ==="
	-sudo ./02_zero_za ; echo "exit=$$?"
	@echo
	@echo "=== probe 3 (sudo): SME i16×i16→i64 matmul ==="
	-sudo ./03_i16_i64_matmul ; echo "exit=$$?"

clean:
	rm -f 01_stream_nop 02_zero_za 03_i16_i64_matmul
