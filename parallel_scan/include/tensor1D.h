#ifndef TENSOR1D_H
#define TENSOR1D_H

#include "unified_memory.h"

#define TENSOR_ACCURACY_EPSILON 0.01

template <typename T>
class Tensor1D : public UnifiedMemory {
private:
    static const char* fmt; // for format string
public:
    unsigned int totalSize;
    T* elements;

    /** 
     * Constructs Tensor1D object with cudaMallocManaged uninitialized elements.
    */
    Tensor1D(unsigned int totalSize);

    /**
     * Constructs Tensor1D object with contents of the specified file.
     * The first row of the file must specify the totalSize of the tensor (vector).
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