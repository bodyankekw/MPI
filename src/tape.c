#include <mpi.h>
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

int main(int argc, char *argv[]) {
    int rank, size;
    double start, end;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int mat_size;
    if (rank == 0) {
        printf("Enter the size of the matrix: ");
        scanf("%d", &mat_size);
    }
    MPI_Bcast(&mat_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    float *matr = NULL;
    float *vec = malloc(mat_size * sizeof(float));
    float *res = malloc(mat_size * sizeof(float));
    int rows_per_proc = mat_size / size;

    float *local_matrix = malloc(rows_per_proc * mat_size * sizeof(float));
    float *local_result = calloc(rows_per_proc, sizeof(float));

    if (rank == 0) {
        matr = malloc(mat_size * mat_size * sizeof(float));
        generate_matrix(&matr, mat_size);
        generate_vector(&vec, mat_size);
        start = MPI_Wtime();
    }

    MPI_Scatter(matr, rows_per_proc * mat_size, MPI_FLOAT, local_matrix, rows_per_proc * mat_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(vec, mat_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < rows_per_proc; ++i)
        for (int j = 0; j < mat_size; ++j)
            local_result[i] += local_matrix[i * mat_size + j] * vec[j];

    MPI_Gather(local_result, rows_per_proc, MPI_FLOAT, res, rows_per_proc, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        end = MPI_Wtime();
        for (int i = 0; i < mat_size; ++i)
            printf("%0.2f\n", res[i]);
        free(matr);


        printf("Total time is %f\n", end - start);
    }

    free(vec);
    free(res);
    free(local_matrix);
    free(local_result);
    MPI_Finalize();
    return 0;
}
