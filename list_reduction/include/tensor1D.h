#ifndef TENSOR1D_H
#define TENSOR1D_H

#include "unified_memory.h"

#define TENSOR_ACCURACY_EPSILON 0.01

class Tensor1D : public UnifiedMemory {
public:
    unsigned int totalSize;
    float* elements;

    /** 
     * Constructs Tensor1D object with cudaMallocManaged uninitialized elements.
    */
    Tensor1D(unsigned int totalSize);

    /**
     * Constructs Tensor1D object with contents of the specified file.
     * The first 3 rows of the file must specify the z, y and x dimensions respectively.
     * @param fileName string specifiying the path to the file to read.
    */
    Tensor1D(char* fileName);

    ~Tensor1D();
    
    void print();

    /**
     * Checks if every element is equal (within TENSOR_ACCURACY_EPSILON for floating point).
     * @param rhs comparison tensor should be equal in each dimension.
    */
    bool operator==(const Tensor1D& rhs);

};

#endif