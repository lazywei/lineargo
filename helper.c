#include <linear.h>
#include <stdio.h>
#include <stdlib.h>

struct feature_node** build_feature_node(double* x, int n_rows, int n_cols, double bias) {
  struct feature_node** fn_x;
  struct feature_node* x_space;
  double elem = 0;
  int n_elems = 0, x_idx = 0, i = 0, j = 0;

  for (i = 0; i < n_rows; i++) {
    for (j = 0; j < n_cols; j++) {
      elem = x[i*n_cols+j];
      if (elem != 0)
        ++n_elems;
    }
  }
  n_elems += n_rows; // for bias term

  fn_x = calloc(n_rows, sizeof(struct feature_node *));
  x_space = calloc(n_elems + n_rows, sizeof(struct feature_node));

  x_idx = 0;
  for (i = 0; i < n_rows; i++) {
    fn_x[i] = &x_space[x_idx];

    for (j = 0; j < n_cols; j++) {
      elem = x[i*n_cols+j];
      if (elem != 0) {
        x_space[x_idx].index = j + 1;
        x_space[x_idx].value = elem;
        ++x_idx;
      }
    }
    if (bias >= 0) {
      x_space[x_idx].index = j + 1;
      x_space[x_idx].value = bias;
    }
    ++x_idx;
    x_space[x_idx].index = -1;
    ++x_idx;
  }

  return fn_x;
}

struct model* call_train(double* x, double* y, int n_rows, int n_cols, double bias,
                         int solver_type, double C, double p, double eps,
                         int nr_weight, int* weight_label, double* weight) {
  int nElems = 0;
  int i, j, x_idx;
  double elem;

  struct feature_node *x_space;
  struct problem prob;
  struct parameter param;

  param.weight_label = weight_label;
  param.weight = weight;
  param.init_sol = NULL;
  param.solver_type = 0;
  param.eps = eps;
  param.C = C;
  param.nr_weight = nr_weight;
  param.p = p;

  prob.bias = bias;
  prob.l = n_rows;
  if (prob.bias >= 0) {
    prob.n = n_cols + 1;
  } else {
    prob.n = n_cols;
  }
  prob.y = y;
  prob.x = build_feature_node(x, n_rows, n_cols, prob.bias);

  return train(&prob, &param);
}


double* call_predict(const struct model *model_, double* x, int n_rows, int n_cols) {
  int i;
  struct feature_node** fn_x;
  double* result;

  result = calloc(n_rows, sizeof(double));

  fn_x = build_feature_node(x, n_rows, n_cols, -1);

  for (i = 0; i < n_rows; ++i) {
    result[i] = predict(model_, fn_x[i]);
  }
  return result;
}

double* call_predict_proba(const struct model *model_, double* x,
                           int n_rows, int n_cols, int n_classes) {
  int i, j;
  struct feature_node** fn_x;
  double* result;
  double* proba;

  result = calloc(n_rows * n_classes, sizeof(double));
  proba = calloc(n_classes, sizeof(double));

  fn_x = build_feature_node(x, n_rows, n_cols, -1);

  for (i = 0; i < n_rows; ++i) {
    predict_probability(model_, fn_x[i], proba);
    for (j = 0; j < n_classes; ++j)
      result[i*n_classes+j] = proba[j];
  }

  free(proba);
  return result;
}
