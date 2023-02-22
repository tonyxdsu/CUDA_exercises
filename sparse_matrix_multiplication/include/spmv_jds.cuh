#ifndef SPMV_JDS_CUH
#define SPMV_JDS_CUH

#include "jds_matrix.h"
#include "tensor1D.h"

#define BLOCK_DIM_JDS 512

template <typename T>
void spmvJDS(const JDSMatrix<T>* mat, const Tensor1D<T>* vec, Tensor1D<T>* res);

#endif