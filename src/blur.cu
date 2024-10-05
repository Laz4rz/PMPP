#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <stdio.h>
#include "cuda_helper.h"

__global__
void blurKernel(unsigned char *img, unsigned char *out, int width, int height, int channels) {
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int column = threadIdx.x + blockDim.x * blockIdx.x;

    int i = row, j = column;
    int caught = 0, r_acc = 0, g_acc = 0, b_acc = 0;
    for (int k=-1; k<=1; k++) {
        for (int l=-1; l<=1; l++) {
            if (i+k >= 0 && i+k < height && j+l >= 0 && j+l < width) {
                r_acc += img[((i+k) * width + (j+l)) * channels];
                g_acc += img[((i+k) * width + (j+l)) * channels + 1];
                b_acc += img[((i+k) * width + (j+l)) * channels + 2];
                caught++;
            }
        }
    } 

    out[(i * width + j) * channels] = r_acc / caught;
    out[(i * width + j) * channels + 1] = g_acc / caught;
    out[(i * width + j) * channels + 2] = b_acc / caught;
}

void blurGPU(unsigned char *img_h, unsigned char *out_h, int width, int height, int channels) {
    int n = width * height * channels;
    int size = n * sizeof(unsigned char);
    unsigned char *img_d, *out_d;

    cudaMallocGuard((void **) &img_d, size);
    cudaMallocGuard((void **) &out_d, size);

    cudaMemcpy(img_d, img_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out_h, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid((width + dimBlock.x - 1.) / dimBlock.x, (height + dimBlock.y- 1.) / dimBlock.y, 1);
    printf("dimGrid: %d, %d, %d\ndimBlock: %d, %d, %d\nTotal threads: %d\n", dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z, dimGrid.x * dimGrid.y * dimBlock.x * dimBlock.y);
    
    blurKernel<<<dimGrid, dimBlock>>>(img_d, out_d, width, height, channels);

    cudaMemcpy(img_h, img_d, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(out_h, out_d, size, cudaMemcpyDeviceToHost);

    cudaFree(img_d);
    cudaFree(out_d);
}

void blurCPU(unsigned char *img, unsigned char *out, int width, int height, int channels) {
    int caught, r_acc, g_acc, b_acc;

    for (int i=0; i<height; i++) {
        for (int j=0; j<width; j++) {
            caught = 0;
            r_acc = 0;
            g_acc = 0;
            b_acc = 0;

            for (int k=-1; k<=1; k++) {
                for (int l=-1; l<=1; l++) {
                    if (i+k >= 0 && i+k < height && j+l >= 0 && j+l < width) {
                        r_acc += img[((i+k) * width + (j+l)) * channels];
                        g_acc += img[((i+k) * width + (j+l)) * channels + 1];
                        b_acc += img[((i+k) * width + (j+l)) * channels + 2];
                        caught++;
                    }
                }
            }

            out[(i * width + j) * channels] = r_acc / caught;
            out[(i * width + j) * channels + 1] = g_acc / caught;
            out[(i * width + j) * channels + 2] = b_acc / caught;
        }
    }
}

int main() {
    // Load an image
    int width, height, channels, size;
    unsigned char *img = stbi_load("assets/Bruegel,_Pieter_de_Oude_-_De_val_van_icarus_-_hi_res.jpg", &width, &height, &channels, 0);
    // unsigned char *img = stbi_load("assets/test_block.jpg", &width, &height, &channels, 0);
    if (img == NULL) {
        printf("Failed to load image!\n");
        return 1;
    }
    size = width * height * channels; 

    unsigned char *out = (unsigned char *)malloc(size * sizeof(unsigned char));
    if (out == NULL) {
            printf("Failed to load image!\n");
            return 1;
    }    

    
    printf("Loaded image with a width of %dpx, height of %dpx, and %d channels\nTotal size:%d\n", width, height, channels, size);

    // Convert the image to grayscale on CPU
    // blurCPU(img, out, width, height, channels);

    // Convert the image to grayscale on GPU
    blurGPU(img, out, width, height, channels);

    // Save the modified image to a new file
    if (stbi_write_png("assets/Breugel_blur.png", width, height, channels, out, width * channels)) {
    // if (stbi_write_png("assets/test_block_blur.png", width, height, channels, img, width * channels)) {
        printf("Image saved successfully\n");
    } else {
        printf("Failed to save image\n");
    }

    stbi_image_free(img);    
    return 0;
}
