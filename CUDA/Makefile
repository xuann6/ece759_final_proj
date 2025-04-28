# Makefile for CUDA RRT Implementations

# CUDA compiler and flags
NVCC = nvcc
# Support multiple GPU architectures - Turing (SM 75), Ampere (SM 80/86)
CUDA_ARCH = -arch=sm_75 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75
NVCC_FLAGS = -std=c++14 -O3 $(CUDA_ARCH)

# Optional profiling flags (uncomment to enable profiling)
#PROFILE_FLAGS = -lineinfo -Xcompiler -rdynamic -Xptxas -v
#NVCC_FLAGS += $(PROFILE_FLAGS)

# Optional debug flags (uncomment for debugging)
#DEBUG_FLAGS = -g -G
#NVCC_FLAGS += $(DEBUG_FLAGS)

# Include directories
INCLUDES = -I.

# Source files
COMMON_SRC = rrtCommon.h cudaRRTUtils.h cudaRRTKernels.h
RRT_SRC = rrtCuda.cu
RRT_STAR_SRC = rrtStarCuda.cu
RRT_BI_SRC = rrtBidirectionalCuda.cu
RRT_INFORMED_SRC = rrtInformedCuda.cu
MAIN_SRC = main.cpp

# Object files
RRT_OBJ = rrtCuda.o
RRT_STAR_OBJ = rrtStarCuda.o
RRT_BI_OBJ = rrtBidirectionalCuda.o
RRT_INFORMED_OBJ = rrtInformedCuda.o
MAIN_OBJ = main.o

# Executables
EXECUTABLE = rrt_cuda

# Phony targets
.PHONY: all clean run ptx profile

# Default target
all: $(EXECUTABLE)

# Main executable
$(EXECUTABLE): $(RRT_OBJ) $(RRT_STAR_OBJ) $(RRT_BI_OBJ) $(RRT_INFORMED_OBJ) $(MAIN_OBJ)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# Compile CUDA source files
$(RRT_OBJ): $(RRT_SRC) $(COMMON_SRC)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

$(RRT_STAR_OBJ): $(RRT_STAR_SRC) $(COMMON_SRC)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

$(RRT_BI_OBJ): $(RRT_BI_SRC) $(COMMON_SRC)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

$(RRT_INFORMED_OBJ): $(RRT_INFORMED_SRC) $(COMMON_SRC)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Compile C++ source files
$(MAIN_OBJ): $(MAIN_SRC)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# PTX generation target
ptx:
	$(NVCC) $(NVCC_FLAGS) -ptx $(RRT_SRC) -o rrtCuda.ptx
	$(NVCC) $(NVCC_FLAGS) -ptx $(RRT_STAR_SRC) -o rrtStarCuda.ptx
	$(NVCC) $(NVCC_FLAGS) -ptx $(RRT_BI_SRC) -o rrtBidirectionalCuda.ptx
	$(NVCC) $(NVCC_FLAGS) -ptx $(RRT_INFORMED_SRC) -o rrtInformedCuda.ptx

# Profiling target
profile: NVCC_FLAGS += -lineinfo -Xcompiler -rdynamic
profile: all
	@echo "Built with profiling flags. Run with nvprof or nsight-compute."

# Clean up
clean:
	rm -f $(RRT_OBJ) $(RRT_STAR_OBJ) $(RRT_BI_OBJ) $(RRT_INFORMED_OBJ) $(MAIN_OBJ) $(EXECUTABLE) *.ptx

# Run the executable
run: $(EXECUTABLE)
	./$(EXECUTABLE)

# Run with specific algorithm
run_standard: $(EXECUTABLE)
	./$(EXECUTABLE) --standard

run_star: $(EXECUTABLE)
	./$(EXECUTABLE) --star

run_bi: $(EXECUTABLE)
	./$(EXECUTABLE) --bi

run_informed: $(EXECUTABLE)
	./$(EXECUTABLE) --informed

# Run without visualization
run_novis: $(EXECUTABLE)
	./$(EXECUTABLE) --no-vis

# Run with custom iterations
run_custom: $(EXECUTABLE)
	./$(EXECUTABLE) --iter 2000 --threads 512

# Run performance test with all algorithms
benchmark: $(EXECUTABLE)
	./$(EXECUTABLE) --no-vis --iter 10000

# Build for multiple GPU architectures
multi_arch:
	$(NVCC) $(NVCC_FLAGS) -gencode=arch=compute_60,code=sm_60 \
		-gencode=arch=compute_70,code=sm_70 \
		-gencode=arch=compute_75,code=sm_75 \
		-gencode=arch=compute_80,code=sm_80 \
		-gencode=arch=compute_86,code=sm_86 \
		-o $(EXECUTABLE) $(RRT_SRC) $(RRT_STAR_SRC) $(RRT_BI_SRC) $(RRT_INFORMED_SRC) $(MAIN_SRC)