#define MAX_FILE_LINE_LENGTH 50
#define MATRIX_EPSILON 0.001

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
