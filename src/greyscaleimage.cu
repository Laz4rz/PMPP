#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <stdio.h>

int main() {
    int width, height, channels;
    unsigned char *img = stbi_load("../assets/Bruegel,_Pieter_de_Oude_-_De_val_van_icarus_-_hi_res.jpg", &width, &height, &channels, 0);
    
    if (img == NULL) {
        printf("Failed to load image!\n");
        return 1;
    }
    
    printf("Loaded image with a width of %dpx, height of %dpx, and %d channels\n", width, height, channels);

    // Don't forget to free the image memory when done
    stbi_image_free(img);
    
    return 0;
}
