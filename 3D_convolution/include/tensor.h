#ifndef TENSOR_H
#define TENSOR_H

#include "unified_memory.h"

class Tensor3D : public UnifiedMemory {
private:
    unsigned int xDim;
    unsigned int yDim;
    unsigned int zDim;
    unsigned int totalSize;
    float* elements;
public:
    /** 
     * Constructs Tensor3D object with cudaMallocManaged uninitialized elements.
     * @param xDim x dimension
     * @param yDim y dimension
     * @param zDim z dimension
    */
    Tensor3D(unsigned int xDim, unsigned int yDim, unsigned int zDim);

    /**
     * Constructs Tensor3D object with contents of the specified file.
     * The first 3 rows of the file must specify the z, y and x dimensions respectively.
     * @param fileName string specifiying the path to the file to read.
    */
    Tensor3D(char* fileName);

    void print();
    // bool operator==(const Tensor3D& rhs);

    // ~Tensor3D();
};

#endif