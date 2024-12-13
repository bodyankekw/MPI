#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


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

void get_block(float *matrix, float *vec,
               int x, int y, int block_size, int mat_size,
               float **block, float **vec_part) {
    *block = (float *) malloc(block_size * block_size * sizeof(float));
    *vec_part = (float *) malloc(block_size * sizeof(float));

    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
            (*block)[get_index(i, j, block_size)] = matrix[get_index(x + i, y + j, mat_size)];
        }
    }
    for (int j = 0; j < block_size; j++) {
        (*vec_part)[j] = vec[y + j];
    }
}

void mat_vec_mul(float *mat, float *vec, int size, float **vec_out) {
    float tmp;
    *vec_out = (float *) malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        tmp = 0.0f;
        for (int j = 0; j < size; j++) {
            tmp += mat[get_index(i, j, size)] * vec[j];
        }
        (*vec_out)[i] = tmp;
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    MPI_Comm cartcomm;
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int grid_size = (int) sqrt((double) size);
    if (grid_size * grid_size != size) {
        if (rank == 0) {
            fprintf(stderr, "Number of processes must be a perfect square!\n");
        }
        MPI_Finalize();
        return 1;
    }

    int mat_size;
    float *matr = NULL;
    float *vec = NULL;
    float *mat_part = NULL;
    float *vec_part = NULL;
    float *res = NULL;
    float *res_reduced = NULL;
    float *tot_res = NULL;
    int block_size;

    if (rank == 0) {
        printf("Enter the size of the matrix: ");
        scanf("%d", &mat_size);
    }
    MPI_Bcast(&mat_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (mat_size % grid_size != 0) {
        if (rank == 0) {
            printf("Matrix size must be divisible by grid size.\n");
        }
        MPI_Finalize();
        return -1;
    }

    int dim[2] = {grid_size, grid_size};
    int period[2] = {0, 0}, reorder = 0;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder, &cartcomm);

    block_size = mat_size / grid_size;

    double start, end;

    if (rank == 0) {
        generate_matrix(&matr, mat_size);
        generate_vector(&vec, mat_size);

        start = MPI_Wtime();

        int coord[2];
        MPI_Cart_coords(cartcomm, 0, 2, coord);

        float *block = NULL;
        get_block(matr, vec, coord[0] * block_size, coord[1] * block_size,
                  block_size, mat_size, &block, &vec_part);

        mat_vec_mul(block, vec_part, block_size, &res);
        free(block);
        free(vec_part);

        for (int i = 1; i < size; i++) {
            MPI_Cart_coords(cartcomm, i, 2, coord);
            float *tmp_block = NULL;
            float *tmp_vec_part = NULL;

            get_block(matr, vec,
                      coord[0] * block_size, coord[1] * block_size,
                      block_size, mat_size, &tmp_block, &tmp_vec_part);

            MPI_Send(tmp_block, block_size * block_size, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            MPI_Send(tmp_vec_part, block_size, MPI_FLOAT, i, 1, MPI_COMM_WORLD);

            free(tmp_block);
            free(tmp_vec_part);
        }
    } else {
        mat_part = (float *) malloc(block_size * block_size * sizeof(float));
        vec_part = (float *) malloc(block_size * sizeof(float));

        MPI_Recv(mat_part, block_size * block_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(vec_part, block_size, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        mat_vec_mul(mat_part, vec_part, block_size, &res);

        free(mat_part);
        free(vec_part);
    }

    int coords[2];
    MPI_Cart_coords(cartcomm, rank, 2, coords);

    MPI_Comm row_comm, col_comm;
    int dims[2];

    dims[0] = 0;
    dims[1] = 1;
    MPI_Cart_sub(cartcomm, dims, &row_comm);


    res_reduced = (float *) malloc(block_size * sizeof(float));
    MPI_Reduce(res, res_reduced, block_size, MPI_FLOAT, MPI_SUM, 0, row_comm);
    free(res);

    dims[0] = 1;
    dims[1] = 0;
    MPI_Cart_sub(cartcomm, dims, &col_comm);

    if (coords[0] == 0) {
        tot_res = (float *) malloc(mat_size * sizeof(float));
    }

    MPI_Gather(res_reduced, block_size, MPI_FLOAT,
               tot_res, block_size, MPI_FLOAT,
               0, col_comm);

    free(res_reduced);

    if (rank == 0) {
        end = MPI_Wtime();


        for (int i = 0; i < mat_size; i++) {
            printf("%0.2f\n", tot_res[i]);
        }

        printf("Total time is %f\n", end - start);

        free(tot_res);
        free(matr);
        free(vec);
    }

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&cartcomm);

    MPI_Finalize();
    return 0;
}