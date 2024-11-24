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

    int N = 4; // Размер матрицы (NxN)
    if (rank == 0) {
        printf("Введите размер матрицы (N): ");
        scanf("%d", &N);
    }
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double *matrix = NULL;
    double *vector = malloc(N * sizeof(double));
    double *result = malloc(N * sizeof(double));
    int rows_per_proc = N / size;

    double *local_matrix = malloc(rows_per_proc * N * sizeof(double));
    double *local_result = calloc(rows_per_proc, sizeof(double));

    if (rank == 0) {
        matrix = malloc(N * N * sizeof(double));
        random_init_matrix_vector(matrix, vector, N, N);
        print_matrix(matrix, N, N);
        print_vector(vector, N);
    }

    MPI_Scatter(matrix, rows_per_proc * N, MPI_DOUBLE, local_matrix, rows_per_proc * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(vector, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < rows_per_proc; ++i)
        for (int j = 0; j < N; ++j)
            local_result[i] += local_matrix[i * N + j] * vector[j];

    MPI_Gather(local_result, rows_per_proc, MPI_DOUBLE, result, rows_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Результат:\n");
        for (int i = 0; i < N; ++i)
            printf("%f\n", result[i]);
        free(matrix);
    }

    free(vector);
    free(result);
    free(local_matrix);
    free(local_result);
    MPI_Finalize();
    return 0;
}
