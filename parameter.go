package liblinear

/*
#cgo LDFLAGS: -llinear
#include <linear.h>
#include <stdio.h>
*/
import "C"

type Parameter struct {
	// struct parameter
	// {
	// 	int solver_type;

	// 	/* these are for training only */
	// 	double eps;	        /* stopping criteria */
	// 	double C;
	// 	int nr_weight;
	// 	int *weight_label;
	// 	double* weight;
	// 	double p;
	// 	double *init_sol;
	// };
	SolverType   int
	Eps, C, P    float64
	ClassWeights map[int]float64
}

func (pm *Parameter) GetPtr() *C.struct_parameter {
	var cParameter C.struct_parameter

	nrWeight := len(pm.ClassWeights)
	weightLabel := []int{}
	weight := []float64{}

	for key, val := range pm.ClassWeights {
		weightLabel = append(weightLabel, key)
		weight = append(weight, val)
	}

	cParameter.solver_type = C.int(pm.SolverType)
	cParameter.eps = C.double(pm.Eps)
	cParameter.C = C.double(pm.C)
	cParameter.p = C.double(pm.P)

	cParameter.nr_weight = C.int(nrWeight)

	if nrWeight > 0 {
		cParameter.weight_label = &mapCInt(weightLabel)[0]
		cParameter.weight = &mapCDouble(weight)[0]
	} else {
		cParameter.weight_label = nil
		cParameter.weight = nil
	}

	return &cParameter
}
