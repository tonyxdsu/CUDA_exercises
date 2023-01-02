#ifndef TENSOR_H
#define TENSOR_H

#include "unified_memory.h"

#define TENSOR_ACCURACY_EPSILON 0.01

class Tensor3D : public UnifiedMemory {
public:
    unsigned int xDim;
    unsigned int yDim;
    unsigned int zDim;
    unsigned int totalSize;
    float* elements;

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

    ~Tensor3D();
    
    void print();

    /**
     * Checks if every element is equal (within TENSOR_ACCURACY_EPSILON for floating point).
     * @param rhs comparison tensor should be equal in each dimension.
    */
    bool operator==(const Tensor3D& rhs);

};

#endif