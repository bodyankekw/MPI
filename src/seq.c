#include <stdio.h>
#include <stdlib.h>

int get_index(int row, int column, int size) {
    return row * size + column;
}

void generate_matrix(float **matrix, int N) {
    *matrix = (float *) malloc(N * N * sizeof(float));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            (*matrix)[get_index(i, j, N)] = (float) (rand() % 10);
        }
    }
}

void generate_vector(float **vec, int vec_size) {
    *vec = (float *) malloc(vec_size * sizeof(float));
    for (int i = 0; i < vec_size; i++) {
        (*vec)[i] = (float) (rand() % 10);
    }
}

void multiply_matrix_vector(float *matrix, float *vector, float *result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = 0;
        for (int j = 0; j < size; j++) {
            result[i] += matrix[get_index(i, j, size)] * vector[j];
        }
    }
}

int main() {
    int mat_size;

    printf("Enter the size of the matrix: ");
    scanf("%d", &mat_size);

    float *matr = NULL;
    float *vec = NULL;
    float *res = malloc(mat_size * sizeof(float));

    generate_matrix(&matr, mat_size);
    generate_vector(&vec, mat_size);


    multiply_matrix_vector(matr, vec, res, mat_size);

    for (int i = 0; i < mat_size; i++) {
        printf("%0.2f\n", res[i]);
    }

    free(matr);
    free(vec);
    free(res);

    return 0;
}
