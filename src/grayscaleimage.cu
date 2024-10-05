#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <stdio.h>

__global__
void grayscaleKernel(unsigned char * img, int size) {
    
}

int main() {
    // Load an image
    int width, height, channels, size;
    unsigned char *img = stbi_load("assets/Bruegel,_Pieter_de_Oude_-_De_val_van_icarus_-_hi_res.jpg", &width, &height, &channels, 0);
    size = width * height * channels; 

    if (img == NULL) {
        printf("Failed to load image!\n");
        return 1;
    }
    
    printf("Loaded image with a width of %dpx, height of %dpx, and %d channels\nTotal size:%d\n", width, height, channels, size);

    // Convert the image to grayscale on CPU
    unsigned char r, g, b, grayscale;
    for (int i=0; i<size; i=i+3) {
        r = img[i];
        g = img[i+1];
        g = img[i+2];

        // Implicit float -> unsigned char conversion 
        grayscale = 0.2126 * r + 0.7152 * g + 0.0722 * b;

        img[i] = grayscale;
        img[i+1] = grayscale;
        img[i+2] = grayscale;
    }

    // Convert the image to grayscale on GPU


    // Save the modified image to a new file
    if (stbi_write_png("assets/Breugel_modified.png", width, height, channels, img, width * channels)) {
        printf("Image saved successfully as output_image.png\n");
    } else {
        printf("Failed to save image\n");
    }

    stbi_image_free(img);    
    return 0;
}
