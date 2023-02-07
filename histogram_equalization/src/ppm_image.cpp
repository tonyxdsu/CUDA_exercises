#include "cuda_runtime.h"

#include <iostream>
#include <fstream>
#include <string.h>
#include <stdio.h>

#include "../include/ppm_image.h"

PPMImage::PPMImage(unsigned int width, unsigned int height) : width(width), height(height) {
    // TODO error checking?
    cudaMallocManaged(&data, width * height * 3 * sizeof(unsigned char));
    cudaDeviceSynchronize();
    // TODO don't need to initialize but will do for testing
    for (int i = 0; i < width * height * 3; i++) {
        data[i] = 0;
    }
}

PPMImage::PPMImage(const std::string& fileName) {
    std::ifstream ifs(fileName, std::ifstream::in | std::ifstream::binary);

    if (!ifs.is_open()) {
        std::cerr << "Unable to open file " << fileName << std::endl;
    }

    // first line
    char curLine[64];
    ifs.getline(curLine, 64);

    if (strcmp(curLine, "P6") != 0) {
        std::cerr << "No P6 tag found in " << fileName << ", got: " << curLine << std::endl;
    }

    // skip comments
    ifs.getline(curLine, 64);
    while (curLine[0] == '#') {
        ifs.getline(curLine, 64);
    }

    // read dimensions
    sscanf(curLine, "%d %d", &width, &height);

    // read maximum pixel value
    ifs.getline(curLine, 64);
    // sscanf(curLine, "%d", &maxPixelValue); // don't think we need this

    // // allocate flattened contiguous memory that is still able to be referenced with A[y][x]
    // rA = new unsigned char*[height];
    // gA = new unsigned char*[height];
    // bA = new unsigned char*[height];

    // // A[0] still hold the entire flattened array so cudamemcpy is happy
    // rA[0] = new unsigned char[height * width];
    // gA[0] = new unsigned char[height * width];
    // bA[0] = new unsigned char[height * width];

    // // point indices A[1] to A[height - 1] to the start of each row so A[y][x] still works
    // for (int y = 1; y < height; y++) {
    // 	rA[y] = rA[y - 1] + width;
    // 	gA[y] = gA[y - 1] + width;
    // 	bA[y] = bA[y - 1] + width;
    // }

    cudaMallocManaged(&data, width * height * 3 * sizeof(unsigned char));
    cudaDeviceSynchronize();
    
    char tempChar;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int index = (y * width + x) * 3;
            ifs.read(&tempChar, 1);
            data[index] = tempChar;

            ifs.read(&tempChar, 1);
            data[index + 1] = tempChar;

            ifs.read(&tempChar, 1);
            data[index + 2] = tempChar;
        }
    }

    ifs.close();
}

PPMImage::~PPMImage() {
    // TODO error checking?
    cudaDeviceSynchronize();
    cudaFree(data);
}

void PPMImage::print() {
    std::cout << width << " " << height << std::endl;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int index = (y * width + x) * 3;
            std::cout << (int)data[index + 0] << " " << (int)data[index + 1] << " " << (int)data[index + 2] << " ";
        }
        std::cout << std::endl;
    }
}

void PPMImage::write(const std::string& fileName) {
    std::ofstream out(fileName, std::ios::binary);
    if (!out) {
        std::cerr << "Error: Failed to open file " << fileName << std::endl;
        exit(1);
    }

    out << "P6" << std::endl;
    out << width << " " << height << std::endl;
    out << 255 << std::endl;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int index = (y * width + x) * 3;
            out << data[index + 0] << data[index + 1] << data[index + 2];
        }
    }

    out.close();
}

bool PPMImage::operator==(const PPMImage& rhs) {
    if (width != rhs.width || height != rhs.height) {
        return false;
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int index = (y * width + x) * 3;
            for (int i = 0; i < 3; i++) {
                if (data[index + i] != rhs.data[index + i]) {
                    return false;
                }
            }
        }
    }

    return true;
}