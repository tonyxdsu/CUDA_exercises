#ifndef JDS_MATRIX_H
#define JDS_MATRIX_H

#include "unified_memory.h"
#include "csr_matrix.h"

#define ACCURACY_EPSILON 0.01

template <typename T>
class JDSMatrix : public UnifiedMemory {
private:
    static const char* fmt; // for format string
    void convertCSRToJDS(T* data, const unsigned int* rowPtr, const unsigned int* colInd, const unsigned int numRows, const unsigned int numDataElements);

public:
    T* data;
    unsigned int* rowPtr;
    unsigned int* colInd;
    unsigned int* rowPerm; 
    unsigned int numRows;
    unsigned int numDataElements;

    /**
     * Constructs JDSMatrix object with contents of the specified file in CSR format.
     * @param dataFileName   File containing the data array.   First element of the file is the total size.
     * @param rowPtrFileName File containing the rowPtr array. First element of the file is the total size.
     * @param colIndFileName File containing the colInd array. First element of the file is the total size.
    */
    // JDSMatrix(char* dataFileName, char* rowPtrFileName, char* colIndFileName);

    /**
     * Constructs JDSMatrix object from CSRMatrix object.
     * O(nlogn) time complexity to sort each row by length.
    */
   JDSMatrix(const CSRMatrix<T>& csrMatrix);


    ~JDSMatrix();
    
    void print();
};

#endif