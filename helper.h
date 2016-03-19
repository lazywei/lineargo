#ifndef HELPER_H_INCLUDED
#define HELPER_H_INCLUDED

#include <linear.h>

struct model* call_train(double* x, double* y, int nCols, int nRows, double bias,
                         int solver_type, double eps, double c, double p,
                         int nr_weight, int* weight_label, double* weight);

#endif
