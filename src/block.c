#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void random_init_matrix_vector(double *matrix, double *vector, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i)
        matrix[i] = rand() % 10;
    for (int i = 0; i < cols; ++i)
        vector[i] = rand() % 10;
}

void print_matrix(double *matrix, int rows, int cols) {
    printf("Матрица:\n");
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%6.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

void print_vector(double *vector, int size) {
    printf("Вектор:\n");
    for (int i = 0; i < size; ++i) {
        printf("%6.2f\n", vector[i]);
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 4; // Размер матрицы
    if (rank == 0) {
        printf("Введите размер матрицы (N): ");
        scanf("%d", &N);
    }
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int grid_size = sqrt(size);
    int block_size = N / grid_size;

    double *matrix = NULL;
    double *vector = malloc(N * sizeof(double));
    double *result = malloc(N * sizeof(double));
    double *local_matrix = malloc(block_size * block_size * sizeof(double));
    double *local_vector = malloc(block_size * sizeof(double));
    double *local_result = calloc(block_size, sizeof(double));

    if (rank == 0) {
        matrix = malloc(N * N * sizeof(double));
        random_init_matrix_vector(matrix, vector, N, N);
        print_matrix(matrix, N, N);
        print_vector(vector, N);
    }

    MPI_Datatype block_type;
    MPI_Type_vector(block_size, block_size, N, MPI_DOUBLE, &block_type);
    MPI_Type_commit(&block_type);

    MPI_Scatter(matrix, 1, block_type, local_matrix, block_size * block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(vector, block_size, MPI_DOUBLE, local_vector, block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < block_size; ++i)
        for (int j = 0; j < block_size; ++j)
            local_result[i] += local_matrix[i * block_size + j] * local_vector[j];

    MPI_Gather(local_result, block_size, MPI_DOUBLE, result, block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Результат:\n");
        for (int i = 0; i < N; ++i)
            printf("%f\n", result[i]);
        free(matrix);
    }

    free(vector);
    free(result);
    free(local_matrix);
    free(local_vector);
    free(local_result);
    MPI_Type_free(&block_type);
    MPI_Finalize();
    return 0;
}
