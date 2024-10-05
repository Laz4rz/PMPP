#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <stdio.h>
#include "cuda_helper.h"

__global__
void grayscaleKernel(unsigned char *img, int width, int height, int channels) {
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int column = threadIdx.x + blockDim.x * blockIdx.x;

    int pixel_idx;
    unsigned char r, g, b, grayscale;
    // printf("Height: %d, Width: %d, row: %d, column: %d\n", height, width, row, column);
    if (row < height && column < width) {
        pixel_idx = (row * width + column) * channels; 
        r = img[pixel_idx];
        g = img[pixel_idx+1];
        b = img[pixel_idx+2];

        grayscale = 0.2126 * r + 0.7152 * g + 0.0722 * b;

        img[pixel_idx] = grayscale;
        img[pixel_idx+1] = grayscale;
        img[pixel_idx+2] = grayscale; 

        // printf("Px %d, %d: %d\n", row, column, grayscale);
    }
}

void grayscaleGPU(unsigned char *img_h, int width, int height, int channels) {
    int n = width * height * channels;
    int size = n * sizeof(unsigned char);
    unsigned char *img_d;

    cudaMallocGuard((void **) &img_d, size);

    cudaMemcpy(img_d, img_h, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(((width * channels) + dimBlock.x - 1.) / dimBlock.x, (height + dimBlock.y- 1.) / dimBlock.y, 1);
    printf("dimGrid: %d, %d, %d\ndimBlock: %d, %d, %d\nTotal threads: %d\n", dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z, dimGrid.x * dimGrid.y * dimBlock.x * dimBlock.y);
    
    grayscaleKernel<<<dimGrid, dimBlock>>>(img_d, width, height, channels);

    cudaMemcpy(img_h, img_d, size, cudaMemcpyDeviceToHost);

    cudaFree(img_d);
}

void grayscaleCPU(unsigned char *img, int size){
    unsigned char r, g, b, grayscale;
    for (int i=0; i<size; i=i+3) {
        r = img[i];
        g = img[i+1];
        b = img[i+2];

        // Implicit float -> unsigned char conversion 
        grayscale = 0.2126 * r + 0.7152 * g + 0.0722 * b;

        img[i] = grayscale;
        img[i+1] = grayscale;
        img[i+2] = grayscale;
    }
}

int main() {
    // Load an image
    int width, height, channels, size;
    unsigned char *img = stbi_load("assets/Bruegel,_Pieter_de_Oude_-_De_val_van_icarus_-_hi_res.jpg", &width, &height, &channels, 0);
    // unsigned char *img = stbi_load("assets/test_block.jpg", &width, &height, &channels, 0);
    size = width * height * channels; 

    if (img == NULL) {
        printf("Failed to load image!\n");
        return 1;
    }
    
    printf("Loaded image with a width of %dpx, height of %dpx, and %d channels\nTotal size:%d\n", width, height, channels, size);

    // Convert the image to grayscale on CPU
    // grayscaleCPU(img, size);

    // Convert the image to grayscale on GPU
    grayscaleGPU(img, width, height, channels);

    // Save the modified image to a new file
    if (stbi_write_png("assets/Breugel_modified.png", width, height, channels, img, width * channels)) {
    // if (stbi_write_png("assets/test_block_modified.png", width, height, channels, img, width * channels)) {
        printf("Image saved successfully\n");
    } else {
        printf("Failed to save image\n");
    }

    stbi_image_free(img);    
    return 0;
}
