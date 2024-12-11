#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int get_index(int row, int column, int size) {
    return row * size + column;
}

void generate_matrix(float** matrix, int N) {
    *matrix = (float *)malloc(N * N * sizeof(float));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            (*matrix)[get_index(i, j, N)] = (float)(rand() % 10); // Random float in range [0, 10)
        }
    }
}

void generate_vector(float** vec, int vec_size) {
    *vec = (float *)malloc(vec_size * sizeof(float));
    for (int i = 0; i < vec_size; i++) {
        (*vec)[i] = (float)(rand() % 10); // Random float in range [0, 10)
    }
}

void get_block(float* matrix, float* vec, int x, int y, int block_size, int mat_size, float** block, float** vec_part) {
    *block = (float *)malloc(block_size * block_size * sizeof(float));
    *vec_part = (float *)malloc(block_size * sizeof(float));

    for (int i = x; i < x + block_size; i++) {
        for (int j = y; j < y + block_size; j++) {
            (*block)[get_index(i - x, j - y, block_size)] = matrix[get_index(i, j, mat_size)];
            (*vec_part)[j - y] = vec[j];
        }
    }
}

void print_block(float* matr, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%f ", matr[get_index(i, j, size)]);
        }
        printf("\n");
    }
}

void mat_vec_mul(float* mat, float* vec, int size, float** vec_out) {
    float tmp;
    *vec_out = (float *)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        tmp = 0;
        for (int j = 0; j < size; j++) {
            tmp += mat[get_index(i, j, size)] * vec[j];
        }
        (*vec_out)[i] = tmp;
    }
}

int main(int argc, char** argv) {
    float* res;
    float* res_reduced;
    float* tot_res;
    int block_size;
    int mat_size;
    float* mat_part;
    float* vec_part;
    float* matr;
    double start, end;
    double gl_start, gl_end;

    MPI_Init(NULL, NULL);

    MPI_Comm cartcomm;


    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int grid_size = (int)sqrt(world_size);

    if (world_rank == 0) {
        printf("Enter the size of the matrix: ");
        scanf("%d", &mat_size);
    }
    MPI_Bcast(&mat_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&grid_size, 1, MPI_INT, 0, MPI_COMM_WORLD);


    if (mat_size % grid_size != 0) {
        if (world_rank == 0) {
            printf("Matrix size must be divisible by grid size.\n");
        }
        MPI_Finalize();
        return -1;
    }

    int dim[2] = {grid_size, grid_size}, period[2] = {0, 0}, reorder = 0;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder, &cartcomm);

    if (world_rank == 0) {
        float* vec;
        generate_matrix(&matr, mat_size);
        generate_vector(&vec, mat_size);
        gl_start = MPI_Wtime();

        float* block;
        int coord[2]={0};
        block_size = mat_size / sqrt(world_size);


        MPI_Cart_coords(cartcomm, 0, 2, coord);
        get_block(matr, vec, coord[0] * block_size, coord[1] * block_size, block_size, mat_size, &block, &vec_part);
        mat_part = block;
        start = MPI_Wtime();
        mat_vec_mul(mat_part, vec_part, block_size, &res);
        end = MPI_Wtime();
        printf("Time to multiply 1 block is %f\n", end - start);
        printf("Process %d has coordinates (%d, %d)\n", world_rank, coord[0], coord[1]);

        start = MPI_Wtime();
        for (int i = 1; i < world_size; i++) {
            MPI_Cart_coords(cartcomm, i, 2, coord);
            printf("Process %d has coordinates (%d, %d)\n", i, coord[0], coord[1]);
            get_block(matr, vec, coord[0] * block_size, coord[1] * block_size, block_size, mat_size, &block, &vec_part);
            MPI_Send(block, block_size * block_size, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&vec[coord[1] * block_size], block_size, MPI_FLOAT, i, 1, MPI_COMM_WORLD);
        }
        end = MPI_Wtime();
        printf("Time to redistribute data is %f\n", end - start);
    }

    MPI_Bcast(&block_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&mat_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    res_reduced = (float *)malloc(block_size * sizeof(float));
    tot_res = (float *)malloc(mat_size * sizeof(float));

    if (world_rank != 0) {
        printf("Multi at proc %d\n", world_rank);
        mat_part = (float *)malloc(block_size * block_size * sizeof(float));
        vec_part = (float *)malloc(block_size * sizeof(float));
        MPI_Recv(mat_part, block_size * block_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(vec_part, block_size, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        mat_vec_mul(mat_part, vec_part, block_size, &res);
        free(mat_part);
        free(vec_part);
    }

    int dims[2] = {0, 1};
    int coord = 0;
    int r;
    MPI_Comm row_comm, col_comm;
    MPI_Cart_sub(cartcomm, dims, &row_comm);
    MPI_Cart_rank(row_comm, &coord, &r);
    MPI_Reduce(res, res_reduced, block_size, MPI_FLOAT, MPI_SUM, r, row_comm);

    dims[0] = 1;
    dims[1] = 0;
    MPI_Cart_sub(cartcomm, dims, &col_comm);
    MPI_Cart_rank(row_comm, &coord, &r);
    MPI_Gather(res_reduced, block_size, MPI_FLOAT, tot_res, block_size, MPI_FLOAT, r, col_comm);

    if (world_rank == 0) {
//        for( int i = 0; i<mat_size; i++)
//        {
//            printf("%f\n",tot_res[i]);
//        }
        gl_end = MPI_Wtime();
        printf("Total time is %f\n", gl_end - gl_start);
        free(tot_res);
        free(res_reduced);
    }

    MPI_Finalize();
}