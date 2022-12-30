#ifndef PARSER_H
#define PARSER_H

#define MAX_FILE_LINE_LENGTH 50
#define ACCURACY_EPSILON 0.01

typedef struct matrix_t {
    unsigned int width;
    unsigned int height;
    float* elements;
} Matrix;

typedef struct tensor3d_t {
    unsigned int xDim;
    unsigned int yDim;
    unsigned int zDim;
    float* elements;
} Tensor3D;

Tensor3D* parseFileToTensor3D(char* inputFileName);
Tensor3D* parseFileToTensor3DCube(char* inputFileName);
void freeTensor(Tensor3D* tensor);
void printTensor(Tensor3D* tensor);
void printMatrix(float* matrix, int height, int width);
int isEqual(float* matrix1, float* matrix2, int height, int width);
int isEqual(Tensor3D* expected, Tensor3D* test);

#endif