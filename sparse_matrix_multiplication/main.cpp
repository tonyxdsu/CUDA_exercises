#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"

#include "include/spmv_csr.cuh"

#include "include/csr_matrix.h"
#include "include/tensor1D.h"

int main(int argc, char** argv) {
    if (argc != 6) {
        printf("Usage: ./executableName dataFileName rowPtrFileName colIndFileName vectorFileName testResultFileName\n");
        return 0;
    }   

    char *dataFileName = argv[1];
    char *rowPtrFileName = argv[2];
    char *colIndFileName = argv[3];
    char *vectorFileName = argv[4];
    char *testResultFileName = argv[5];

    CSRMatrix<float>* csrMat = new CSRMatrix<float>(dataFileName, rowPtrFileName, colIndFileName);
    Tensor1D<float>* vec = new Tensor1D<float>(vectorFileName);
    Tensor1D<float>* res = new Tensor1D<float>(csrMat->numRows);
    Tensor1D<float>* testRes = new Tensor1D<float>(testResultFileName);

    spmvCSR<float>(csrMat, vec, res);

    if (*res == *testRes) {
        printf("Test passed!\n");
    } else {
        printf("Test failed!\n");
    }

    delete csrMat;
    delete vec;
    delete res;
    delete testRes;

    return 0;
}