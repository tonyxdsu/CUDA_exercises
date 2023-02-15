#ifndef CSR_MATRIX_H
#define CSR_MATRIX_H

#include "unified_memory.h"

#define ACCURACY_EPSILON 0.01

template <typename T>
class CSRMatrix : public UnifiedMemory {
private:
    static const char* fmt; // for format string
public:
    T* data;
    unsigned int* rowPtr;
    unsigned int* colInd;
    unsigned int numRows;
    
    /**
     * Constructs CSRMatrix object with contents of the specified file in CSR format.
     * @param dataFileName   File containing the data array.   First element of the file is the total size.
     * @param rowPtrFileName File containing the rowPtr array. First element of the file is the total size.
     * @param colIndFileName File containing the colInd array. First element of the file is the total size.
    */
    CSRMatrix(char* dataFileName, char* rowPtrFileName, char* colIndFileName);

    ~CSRMatrix();
    
    void print();
};

#endif