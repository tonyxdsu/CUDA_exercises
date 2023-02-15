#ifndef SPMV_CSR_CUH
#define SPMV_CSR_CUH

#include "csr_matrix.h"
#include "tensor1D.h"

#define BLOCK_DIM_CSR 512

template <typename T>
void spmvCSR(const CSRMatrix<T>* mat, const Tensor1D<T>* vec, Tensor1D<T>* res);

#endif