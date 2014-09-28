SPARSECODING_DIR := $(shell readlink $(dir $(lastword $(MAKEFILE_LIST))) -f)
PETUUM_ROOT = $(SPARSECODING_DIR)/../../

include $(PETUUM_ROOT)/defns.mk

SPARSECODING_SRC = $(wildcard $(SPARSECODING_DIR)/src/*.cpp)
SPARSECODING_HDR = $(wildcard $(SPARSECODING_DIR)/src/*.hpp)
TOOLS_SRC = $(wildcard $(SPARSECODING_DIR)/src/tools/*.cpp)
TOOLS_HDR = $(wildcard $(SPARSECODING_DIR)/src/tools/*.hpp)
SPARSECODING_BIN = $(SPARSECODING_DIR)/bin
SPARSECODING_OBJ = $(SPARSECODING_SRC:.cpp=.o)
TOOLS_OBJ= $(TOOLS_SRC:.cpp=.o)

all: sparsecoding_main

sparsecoding_main: $(SPARSECODING_BIN)/sparsecoding_main

$(SPARSECODING_BIN):
	mkdir -p $(SPARSECODING_BIN)

$(SPARSECODING_BIN)/sparsecoding_main: $(SPARSECODING_OBJ) $(TOOLS_OBJ) $(PETUUM_PS_LIB) $(SPARSECODING_BIN)
	$(PETUUM_CXX) $(PETUUM_CXXFLAGS) $(PETUUM_INCFLAGS) \
	$(SPARSECODING_OBJ) $(TOOLS_OBJ) $(PETUUM_PS_LIB) $(PETUUM_LDFLAGS) -o $@

$(SPARSECODING_OBJ): %.o: %.cpp $(SPARSECODING_HDR) $(TOOLS_HDR)
	$(PETUUM_CXX) $(PETUUM_CXXFLAGS) -Wno-unused-result $(PETUUM_INCFLAGS) -c $< -o $@

$(TOOLS_OBJ): %.o: %.cpp $(TOOLS_HDR)
	$(PETUUM_CXX) $(PETUUM_CXXFLAGS) -Wno-unused-result $(PETUUM_INCFLAGS) -c $< -o $@

clean:
	rm -rf $(SPARSECODING_OBJ)
	rm -rf $(TOOLS_OBJ)
	rm -rf $(SPARSECODING_BIN)

.PHONY: clean sparsecoding_main
