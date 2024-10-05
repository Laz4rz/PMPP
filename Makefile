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

# Source files for each target
SRCS1 = $(SRC_DIR)/vecadd.cu $(LIB_DIR)/cuda_helper.cu
SRCS2 = $(SRC_DIR)/grayscale.cu $(LIB_DIR)/cuda_helper.cu
SRCS3 = $(SRC_DIR)/blur.cu $(LIB_DIR)/cuda_helper.cu

# Include directories
INCLUDES = -I$(INCLUDE_DIR)

# Default target: Build both executables
all: $(TARGET1) $(TARGET2) $(TARGET3)

# Rule to build the first executable (vecadd)
$(TARGET1): $(SRCS1)
	$(NVCC) $(INCLUDES) -o $(TARGET1) $(SRCS1)

# Rule to build the second executable (greyscaleimage)
$(TARGET2): $(SRCS2)
	$(NVCC) $(INCLUDES) -o $(TARGET2) $(SRCS2)

# Rule to build the third executable (blurimage)
$(TARGET3): $(SRCS3)
	$(NVCC) $(INCLUDES) -o $(TARGET3) $(SRCS3)

# Clean target: Remove both executables
clean:
	rm -f $(TARGET1) $(TARGET2) $(TARGET3)
	rm -f assets/Breugel_grayscale.png
	rm -f assets/Breugel_blur.png