package lineargo

/*
#cgo LDFLAGS: -llinear
#include <linear.h>
#include <stdio.h>
#include "helper.h"
*/
import "C"

import (
	"errors"
	"fmt"

	"github.com/gonum/matrix/mat64"
)

// Model contains a pointer to C's struct model (i.e., `*C.struct_model`). It is
// returned after training and used for predicting.
type Model struct {
	// struct model
	// {
	// 	struct parameter param;
	// 	int nr_class;		/* number of classes */
	// 	int nr_feature;
	// 	double *w;
	// 	int *label;		/* label of each class */
	// 	double bias;
	// };
	cModel *C.struct_model
}

// Wrapper for the `train` function in liblinear.
//
// `model* train(const struct problem *prob, const struct parameter *param);`
//
// The explanation of parameters are:
//
// solverType:
//
//   for multi-class classification
//          0 -- L2-regularized logistic regression (primal)
//          1 -- L2-regularized L2-loss support vector classification (dual)
//          2 -- L2-regularized L2-loss support vector classification (primal)
//          3 -- L2-regularized L1-loss support vector classification (dual)
//          4 -- support vector classification by Crammer and Singer
//          5 -- L1-regularized L2-loss support vector classification
//          6 -- L1-regularized logistic regression
//          7 -- L2-regularized logistic regression (dual)
//   for regression
//         11 -- L2-regularized L2-loss support vector regression (primal)
//         12 -- L2-regularized L2-loss support vector regression (dual)
//         13 -- L2-regularized L1-loss support vector regression (dual)
//
// eps is the stopping criterion.
//
// C_ is the cost of constraints violation.
//
// p is the sensitiveness of loss of support vector regression.
//
// classWeights is a map from int to float64, with the key be the class and the
// value be the weight. For example, {1: 10, -1: 0.5} means giving weight=10 for
// class=1 while weight=0.5 for class=-1
//
// If you do not want to change penalty for any of the classes, just set
// classWeights to nil.
func Train(X, y *mat64.Dense, bias float64, solverType int, c_, p, eps float64, classWeights map[int]float64) *Model {
	var weightLabelPtr *C.int
	var weightPtr *C.double

	nRows, nCols := X.Dims()

	cX := mapCDouble(X.RawMatrix().Data)
	cY := mapCDouble(y.ColView(0).RawVector().Data)

	nrWeight := len(classWeights)
	weightLabel := []C.int{}
	weight := []C.double{}

	for key, val := range classWeights {
		weightLabel = append(weightLabel, (C.int)(key))
		weight = append(weight, (C.double)(val))
	}

	if nrWeight > 0 {
		weightLabelPtr = &weightLabel[0]
		weightPtr = &weight[0]
	} else {
		weightLabelPtr = nil
		weightPtr = nil
	}

	model := C.call_train(
		&cX[0], &cY[0],
		C.int(nRows), C.int(nCols), C.double(bias),
		C.int(solverType), C.double(c_), C.double(p), C.double(eps),
		C.int(nrWeight), weightLabelPtr, weightPtr)

	return &Model{
		cModel: model,
	}
}

// double predict(const struct model *model_, const struct feature_node *x);
func Predict(model *Model, X *mat64.Dense) *mat64.Dense {
	nRows, nCols := X.Dims()
	cX := mapCDouble(X.RawMatrix().Data)
	y := mat64.NewDense(nRows, 1, nil)
	result := doubleToFloats(C.call_predict(
		model.cModel, &cX[0], C.int(nRows), C.int(nCols)), nRows)
	y.SetCol(0, result)
	return y
}

// double predict_probability(const struct model *model_, const struct feature_node *x, double* prob_estimates);
func PredictProba(model *Model, X *mat64.Dense) *mat64.Dense {
	nRows, nCols := X.Dims()
	nrClasses := int(C.get_nr_class(model.cModel))

	cX := mapCDouble(X.RawMatrix().Data)
	y := mat64.NewDense(nRows, nrClasses, nil)

	result := doubleToFloats(C.call_predict_proba(
		model.cModel, &cX[0], C.int(nRows), C.int(nCols), C.int(nrClasses)),
		nRows*nrClasses)
	for i := 0; i < nRows; i++ {
		y.SetRow(i, result[i*nrClasses:(i+1)*nrClasses])
	}
	return y
}

func Accuracy(y_true, y_pred *mat64.Dense) float64 {
	y1 := y_true.ColView(0).RawVector().Data
	y2 := y_pred.ColView(0).RawVector().Data

	total := 0.0
	correct := 0.0

	for i := 0; i < len(y1); i++ {
		if y1[i] == y2[i] {
			correct++
		}
		total++
	}
	return correct / total
}

func SaveModel(model *Model, filename string) {
	rtn := C.save_model(C.CString(filename), model.cModel)
	if int(rtn) != 0 {
		errStr := fmt.Sprintf("Error Code `%v` when trying to save model", int(rtn))
		fmt.Println(errStr)
		panic(errors.New(errStr))
	}
}

func LoadModel(filename string) *Model {
	model := C.load_model(C.CString(filename))
	if model == nil {
		errStr := fmt.Sprintf("Can't load model from %v", filename)
		panic(errors.New(errStr))
	}
	return &Model{
		cModel: model,
	}
}
