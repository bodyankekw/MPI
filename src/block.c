#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void print_matrix(float *matrix, int rows, int cols) {
    printf("Матрица:\n");
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%6.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

void print_vector(float *vector, int size) {
    printf("Вектор:\n");
    for (int i = 0; i < size; ++i) {
        printf("%6.2f\n", vector[i]);
    }
}

int get_index(int row, int column, int size) {
    return row * size + column;
}

void generate_matrix(float** matrix, int mat_size) {
    *matrix = (float*)malloc(mat_size * mat_size * sizeof(float));
    for (int i = 0; i < mat_size; i++) {
        for (int j = 0; j < mat_size; j++) {
            (*matrix)[get_index(i, j, mat_size)] = (float)(rand() % 10); // Random int [0-9]
        }
    }
}

void generate_vector(float** vec, int vec_size) {
    *vec = (float*)malloc(vec_size * sizeof(float));
    for (int i = 0; i < vec_size; i++) {
        (*vec)[i] = (float)(rand() % 10); // Random int [0-9]
    }
}

void mat_vec_mul(float* mat, float* vec, int rows, int cols, float* res) {
    for (int i = 0; i < rows; i++) {
        res[i] = 0;
        for (int j = 0; j < cols; j++) {
            res[i] += mat[get_index(i, j, cols)] * vec[j];
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int mat_size = 4; // Размер матрицы

    if (world_rank == 0) {
        printf("Введите размер матрицы (N): ");
        scanf("%d", &mat_size);
    }
    MPI_Bcast(&mat_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int block_rows = mat_size / world_size; // Число строк на процесс
    if (mat_size % world_size != 0) {
        if (world_rank == 0) {
            printf("Error: Matrix size must be divisible by the number of processes.\n");
        }
        MPI_Finalize();
        return -1;
    }

    float* matrix = NULL;
    float* vector = NULL;
    float* local_matrix = (float*)malloc(block_rows * mat_size * sizeof(float));
    float* local_result = (float*)malloc(block_rows * sizeof(float));
    matrix = (float*)malloc(mat_size * mat_size * sizeof(float));
    vector = (float*)malloc(mat_size * sizeof(float));
    if (world_rank == 0) {
        // Генерация матрицы и вектора
        generate_matrix(&matrix, mat_size);
        generate_vector(&vector, mat_size);
        //print_matrix(matrix, mat_size, mat_size);
        //print_vector(vector, mat_size);
        printf("\n");
    }

    // Передача данных: делим матрицу и вектор
    MPI_Scatter(matrix, block_rows * mat_size, MPI_FLOAT, local_matrix, block_rows * mat_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(vector, mat_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Умножение локальной части матрицы на вектор
    mat_vec_mul(local_matrix, vector, block_rows, mat_size, local_result);

    // Сбор результатов
    float* result = NULL;
    if (world_rank == 0) {
        result = (float*)malloc(mat_size * sizeof(float));
    }
    MPI_Gather(local_result, block_rows, MPI_FLOAT, result, block_rows, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        print_vector(result,mat_size);
        printf("\n");
        free(result);
    }

    free(local_matrix);
    free(local_result);
    if (world_rank == 0) {
        free(matrix);
        free(vector);
    }

    MPI_Finalize();
    return 0;
}