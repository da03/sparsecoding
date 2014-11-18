SPARSECODING_DIR := $(shell readlink $(dir $(lastword $(MAKEFILE_LIST))) -f)
PETUUM_ROOT = $(SPARSECODING_DIR)/../../

include $(PETUUM_ROOT)/defns.mk

SPARSECODING_SRC = $(wildcard $(SPARSECODING_DIR)/src/*.cpp)
SPARSECODING_HDR = $(wildcard $(SPARSECODING_DIR)/src/*.hpp)
UTIL_SRC = $(wildcard $(SPARSECODING_DIR)/src/util/*.cpp)
UTIL_HDR = $(wildcard $(SPARSECODING_DIR)/src/util/*.hpp)
SPARSECODING_BIN = $(SPARSECODING_DIR)/bin
SPARSECODING_OBJ = $(SPARSECODING_SRC:.cpp=.o)
UTIL_OBJ= $(UTIL_SRC:.cpp=.o)
# PETUUM_CXXFLAGS:=$(PETUUM_CXXFLAGS) -fopenmp

)all: sparsecoding_main

sparsecoding_main: $(SPARSECODING_BIN)/sparsecoding_main

$(SPARSECODING_BIN):
	mkdir -p $(SPARSECODING_BIN)

$(SPARSECODING_BIN)/sparsecoding_main: $(SPARSECODING_OBJ) $(UTIL_OBJ) $(PETUUM_PS_LIB) $(SPARSECODING_BIN)
	$(PETUUM_CXX) $(PETUUM_CXXFLAGS) $(PETUUM_INCFLAGS) \
	$(SPARSECODING_OBJ) $(UTIL_OBJ) $(PETUUM_PS_LIB) $(PETUUM_LDFLAGS) -o $@

$(SPARSECODING_OBJ): %.o: %.cpp $(SPARSECODING_HDR) $(UTIL_HDR)
	$(PETUUM_CXX) $(PETUUM_CXXFLAGS) -Wno-unused-result $(PETUUM_INCFLAGS) -c $< -o $@

$(UTIL_OBJ): %.o: %.cpp $(UTIL_HDR)
	$(PETUUM_CXX) $(PETUUM_CXXFLAGS) -Wno-unused-result $(PETUUM_INCFLAGS) -c $< -o $@

clean:
	rm -rf $(SPARSECODING_OBJ)
	rm -rf $(UTIL_OBJ)
	rm -rf $(SPARSECODING_BIN)

.PHONY: clean sparsecoding_main
