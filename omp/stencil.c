#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define NROWS 2000
#define NCOLS 2000
#define NSTEPS 5000

void stencil_step(float **in, float **out, int rows, int cols) {
#pragma omp parallel for
  for (int i = 1; i < rows - 1; i++) {
#pragma omp simd    
    for (int j = 1; j < cols - 1; j++) {
      out[i][j] = 0.2f * (in[i][j] +
                  in[i-1][j] + in[i+1][j] +
                  in[i][j-1] + in[i][j+1]);
    }
  }
}

float **alloc_grid(int rows, int cols) {
  float **grid = malloc(rows * sizeof(float *));
  float *data = aligned_alloc(32, rows * cols * sizeof(float));
  for (int i = 0; i < rows; i++) {
    grid[i] = &data[i * cols];
  }
  return grid;
}

void init_grid(float **grid, int rows, int cols) {
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
      grid[i][j] = (i == rows/2 && j == cols/2) ? 100.0f : 0.0f;
}

int main() {
  float **A = alloc_grid(NROWS, NCOLS);
  float **B = alloc_grid(NROWS, NCOLS);

  init_grid(A, NROWS, NCOLS);

  double start = omp_get_wtime();

  for (int t = 0; t < NSTEPS; t++) {
    stencil_step(A, B, NROWS, NCOLS);
    float **tmp = A; A = B; B = tmp;
  }

  double end = omp_get_wtime();
  printf("Time: %.3f seconds\n", end - start);
  printf("Final center value: %.2f\n", A[NROWS/2][NCOLS/2]);

  free(A[0]); free(A);
  free(B[0]); free(B);

  return 0;
}
