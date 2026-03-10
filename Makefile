NVCC = nvcc
ARCH ?= sm_80
FLAGS = -O3 -arch=$(ARCH) -Iinclude

all: galaxy_merger

galaxy_merger: src/gravity.cu src/main.cu include/galaxy_types.h
	$(NVCC) $(FLAGS) src/gravity.cu src/main.cu -o galaxy_merger

clean:
	rm -f galaxy_merger
