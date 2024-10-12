# Compiler
NVCC = nvcc

# Directories
SRC_DIR = src
INCLUDE_DIR = include
LIB_DIR = lib

# Output executable names
TARGET1 = vecadd
TARGET2 = grayscale
TARGET3 = blur
TARGET4 = matmul_naive
TARGET5 = matmul_tiled

# Source files for each target
SRCS1 = $(SRC_DIR)/vecadd.cu $(LIB_DIR)/cuda_helper.cu
SRCS2 = $(SRC_DIR)/grayscale.cu $(LIB_DIR)/cuda_helper.cu
SRCS3 = $(SRC_DIR)/blur.cu $(LIB_DIR)/cuda_helper.cu
SRCS4 = $(SRC_DIR)/matmul_naive.cu $(LIB_DIR)/cuda_helper.cu
SRCS5 = $(SRC_DIR)/matmul_tiled.cu $(LIB_DIR)/cuda_helper.cu

# Include directories
INCLUDES = -I$(INCLUDE_DIR)

# Default target: Build all executables
all: $(TARGET1) $(TARGET2) $(TARGET3) $(TARGET4) $(TARGET5)

# Rule to build the first executable (vecadd)
$(TARGET1): $(SRCS1)
	$(NVCC) $(INCLUDES) -o $(TARGET1) $(SRCS1)

# Rule to build the second executable (grayscale)
$(TARGET2): $(SRCS2)
	$(NVCC) $(INCLUDES) -o $(TARGET2) $(SRCS2)

# Rule to build the third executable (blur)
$(TARGET3): $(SRCS3)
	$(NVCC) $(INCLUDES) -o $(TARGET3) $(SRCS3)

# Rule to build the fourth executable (matmul_naive)
$(TARGET4): $(SRCS4)
	$(NVCC) $(INCLUDES) -o $(TARGET4) $(SRCS4)

# Rule to build the fifth executable (matmul_tiled)
$(TARGET5): $(SRCS5)
	$(NVCC) $(INCLUDES) -o $(TARGET5) $(SRCS5)

# PTX target: Generate PTX for all sources
ptx: $(SRCS1) $(SRCS2) $(SRCS3) $(SRCS4) $(SRCS5)
	$(NVCC) $(INCLUDES) -ptx -o $(SRC_DIR)/vecadd.ptx $(SRCS1)
	$(NVCC) $(INCLUDES) -ptx -o $(SRC_DIR)/grayscale.ptx $(SRCS2)
	$(NVCC) $(INCLUDES) -ptx -o $(SRC_DIR)/blur.ptx $(SRCS3)
	$(NVCC) $(INCLUDES) -ptx -o $(SRC_DIR)/matmul_naive.ptx $(SRCS4)
	$(NVCC) $(INCLUDES) -ptx -o $(SRC_DIR)/matmul_tiled.ptx $(SRCS5)

# SASS target: Generate SASS assembly (requires --keep flag)
sass: $(TARGET1) $(TARGET2) $(TARGET3) $(TARGET4) $(TARGET5)
	cuobjdump --dump-sass $(TARGET1) > $(TARGET1).sass
	cuobjdump --dump-sass $(TARGET2) > $(TARGET2).sass
	cuobjdump --dump-sass $(TARGET3) > $(TARGET3).sass
	cuobjdump --dump-sass $(TARGET4) > $(TARGET4).sass
	cuobjdump --dump-sass $(TARGET5) > $(TARGET5).sass

# Clean target: Remove executables and intermediate files
clean:
	rm -f $(TARGET1) $(TARGET2) $(TARGET3) $(TARGET4) $(TARGET5)
	rm -f assets/Breugel_grayscale.png assets/Breugel_blur.png
	rm -f $(SRC_DIR)/*.ptx *.sass
