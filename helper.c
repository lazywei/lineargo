#include <linear.h>
#include <stdio.h>

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

struct model* call_train(double* x, double* y, int nCols, int nRows, double bias, struct parameter param) {
  int nElems = 0;
  int i, j, x_idx;
  double elem;

  struct feature_node *x_space;
  struct problem prob;
  struct model* model_;

  // printf("\nnCols= %d\n", nCols);
  // printf("\nnRows= %d\n", nRows);

  prob.bias = bias;
  prob.l = nRows;
  if (prob.bias >= 0) {
    prob.n = nCols + 1;
  } else {
    prob.n = nCols;
  }
  prob.y = y;

  for (i = 0; i < nRows; i++) {
    for (j = 0; j < nCols; j++) {
      elem = x[i*nCols+j];
      if (elem != 0) {
        ++nElems;
      }
    }
    nElems++; // for bias term
  }
  // printf("\n\nnElems = %d\n", nElems);

  prob.x = Malloc(struct feature_node *, nRows);
  x_space = Malloc(struct feature_node, nElems + nRows);

  x_idx = 0;
  for (i = 0; i < nRows; i++) {
    prob.x[i] = &x_space[x_idx];

    for (j = 0; j < nCols; j++) {
      elem = x[i*nCols+j];
      if (elem != 0) {
        x_space[x_idx].index = j + 1;
        x_space[x_idx].value = elem;
        ++x_idx;
      }
    }
    if (prob.bias >= 0) {
      x_space[x_idx].index = j + 1;
      x_space[x_idx].value = prob.bias;
    }
    ++x_idx;
    x_space[x_idx].index = -1;
    ++x_idx;
  }
  // printf("\nx_idx = %d\n", x_idx);

  return train(&prob, &param);
}
