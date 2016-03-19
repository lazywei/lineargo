#ifndef HELPER_H_INCLUDED
#define HELPER_H_INCLUDED

#include <linear.h>

struct model* call_train(double* x, double* y, int nCols, int nRows, double bias,
                         int solver_type, double c, double p, double eps,
                         int nr_weight, int* weight_label, double* weight);

double* call_predict(const struct model *model_, const double* x, int nCols, int nRows);

double* call_predict_proba(const struct model *model_, double* x,
                           int n_rows, int n_cols, int n_classes);

#endif
