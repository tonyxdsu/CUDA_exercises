#define MAX_FIRST_LINE_LENGTH 50
#define MATRIX_EPSILON 0.001

float* parseFileToMatrix(char* inputFileName, int* height, int* width);
void printMatrix(float* matrix, int height, int width);
int isEqual(float* matrix1, float* matrix2, int height, int width);
