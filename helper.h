#ifndef HELPER_H_INCLUDED
#define HELPER_H_INCLUDED

#include <linear.h>

struct model* call_train(double* x, double* y, int nCols, int nRows, double bias, struct parameter param);

#endif
