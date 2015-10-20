CUDA_PATH := /usr/local/cuda
OPENBLAS_PATH := $(shell brew --prefix openblas)
CC_FLAGS := -DARMA_USE_BLAS -larmadillo -lopenblas -L$(OPENBLAS_PATH)/lib
BINDIR := ./bin
SRCDIR := ./src
TARGET := ann
CC := g++-5

$(shell mkdir -p $(BINDIR))

clean:
	@rm -f $(BINDIR)/*

build: clean
	@$(CC) $(CC_FLAGS) $(SRCDIR)/**/*.cpp $(SRCDIR)/main.cpp -o ./bin/$(TARGET)

build_cuda: clean
	@$(CC) -lnvblas -I$(CUDA_PATH)/include -L$(CUDA_PATH)/lib $(CC_FLAGS) $(SRCDIR)/**/*.cpp $(SRCDIR)/main.cpp -o ./bin/$(TARGET)

run: build
	@./bin/$(TARGET)

run_cuda: build_cuda
	@./bin/$(TARGET)
