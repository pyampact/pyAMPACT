// dpcore_wrapper.c
// Wrapper to expose dpcore as a shared library for Python

#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

#define INF DBL_MAX

// Core DTW computation (stripped from dpcore.c)
void dpcore_compute(const double* S, int rows, int cols,
                    const double* C, int crows, int ccols,
                    double* D, double* P) {

    for (int i = 0; i < rows * cols; ++i) {
        D[i] = INF;
        P[i] = -1;
    }

    D[0] = S[0];

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int idx = j * rows + i;
            double min_cost = INF;
            int min_step = -1;
            for (int k = 0; k < crows; ++k) {
                int di = (int)C[k];
                int dj = (int)C[k + crows];
                double cost = C[k + 2 * crows];

                int ii = i - di;
                int jj = j - dj;
                if (ii >= 0 && jj >= 0) {
                    int prev_idx = jj * rows + ii;
                    double val = D[prev_idx] + S[idx] * cost;
                    if (val < min_cost) {
                        min_cost = val;
                        min_step = k;
                    }
                }
            }
            if (i > 0 || j > 0) {
                D[idx] = min_cost;
                P[idx] = min_step;
            }
        }
    }
}

// Shared library function for Python
__attribute__((visibility("default")))
int run_dpcore(const double* S, int rows, int cols,
               const double* C, int crows, int ccols,
               double* D, double* P) {
    if (!S || !C || !D || !P || rows <= 0 || cols <= 0) return -1;
    dpcore_compute(S, rows, cols, C, crows, ccols, D, P);
    return 0;
}