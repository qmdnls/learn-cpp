# ---- config -------------------------------------------------------

CXX           ?= clang++               # or g++
CXXFLAGS      ?= -std=c++17 -O2 -Wall

# home‑brew on intel mac: /usr/local/opt/openblas
# home‑brew on apple‑silicon: /opt/homebrew/opt/openblas
# ubuntu/debian: leave empty and rely on /usr/lib inclusion
OPENBLAS_ROOT ?= /usr/local/opt/openblas

INCLUDE_FLAGS := -I$(OPENBLAS_ROOT)/include
LIB_FLAGS     := -L$(OPENBLAS_ROOT)/lib -lopenblas

# ------------------------------------------------------------------

OBJ  = tensor.o
DEMOS = mlp_train

all: $(DEMOS)

tensor.o: tensor.cpp tensor.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) -c $< -o $@

mlp_train: mlp_train.cpp $(OBJ) tensor.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) $< $(OBJ) $(LIB_FLAGS) -o $@

clean:
	rm -f $(OBJ) $(DEMOS)

.PHONY: all clean
