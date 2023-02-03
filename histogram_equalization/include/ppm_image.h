#ifndef PPM_IMAGE_H
#define PPM_IMAGE_H

#include <string>
#include "unified_memory.h"

class PPMImage : public UnifiedMemory {
public:
    unsigned int width;
    unsigned int height;
    unsigned char* data;

    /** 
     * Constructs PPMImage object with specified width and height with unitialized data.
    */
    PPMImage(unsigned int width, unsigned int height);

    /**
     * Constructs PPMImage object with data from specified file.
     * @param fileName string specifiying the path to the file to read.
    */
    PPMImage(const std::string& fileName);

    ~PPMImage();
    
    void print();

    void write(const std::string& fileName);

    /**
     * Checks if every element is equal (within TENSOR_ACCURACY_EPSILON for floating point).
     * @param rhs comparison tensor should be equal in each dimension.
    */
    bool operator==(const PPMImage& rhs);
};

#endif