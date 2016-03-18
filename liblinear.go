package liblinear

/*
#cgo LDFLAGS: -llinear
#include <linear.h>
#include <stdio.h>
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

static struct model* call_train(double* x, double* y, int nCols, int nRows, double bias, struct parameter param) {
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
*/
import "C"

import (
	"errors"
	"fmt"
	"unsafe"

	"github.com/gonum/matrix/mat64"
)

type FeatureNode struct {
	// struct feature_node
	// {
	// 	int index;
	// 	double value;
	// };
	cFeatureNode *C.struct_feature_node
}

func NewFeatureNode(index int, value float64) *FeatureNode {
	return &FeatureNode{
		cFeatureNode: &C.struct_feature_node{
			index: C.int(index),
			value: C.double(value),
		},
	}
}

func (fn *FeatureNode) GetData() (int, float64) {
	return int(fn.cFeatureNode.index), float64(fn.cFeatureNode.value)
}

func (fn *FeatureNode) GetPtr() *C.struct_feature_node {
	return fn.cFeatureNode
}

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

func toFeatureNodes(X *mat64.Dense) []*C.struct_feature_node {
	featureNodes := []*C.struct_feature_node{}

	nRows, nCols := X.Dims()

	for i := 0; i < nRows; i++ {
		row := []C.struct_feature_node{}
		for j := 0; j < nCols; j++ {
			val := X.At(i, j)
			if val != 0 {
				row = append(row, C.struct_feature_node{
					index: C.int(j + 1),
					value: C.double(val),
				})
			}
		}

		row = append(row, C.struct_feature_node{
			index: C.int(-1),
			value: C.double(0),
		})
		featureNodes = append(featureNodes, &row[0])
	}

	return featureNodes
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
func Train(X, y *mat64.Dense, bias float64, pm *Parameter) *Model {

	// var problem C.struct_problem

	nRows, nCols := X.Dims()

	cY := y.ColView(0).RawVector().Data
	cX := X.RawMatrix().Data

	// model := C.call_train(problem, pm.GetPtr())
	model := C.call_train((*C.double)(&cX[0]), (*C.double)(unsafe.Pointer(&cY[0])), C.int(nCols), C.int(nRows), C.double(bias), pm.GetPtr())
	return &Model{
		cModel: model,
	}
}

// double predict(const struct model *model_, const struct feature_node *x);
func Predict(model *Model, X *mat64.Dense) *mat64.Dense {
	nRows, _ := X.Dims()
	cXs := toFeatureNodes(X)
	y := mat64.NewDense(nRows, 1, nil)
	for i, cX := range cXs {
		y.Set(i, 0, float64(C.predict(model.cModel, cX)))
	}
	return y
}

// double predict_probability(const struct model *model_, const struct feature_node *x, double* prob_estimates);
func PredictProba(model *Model, X *mat64.Dense) *mat64.Dense {
	nRows, _ := X.Dims()
	nrClasses := int(C.get_nr_class(model.cModel))

	cXs := toFeatureNodes(X)
	y := mat64.NewDense(nRows, nrClasses, nil)

	proba := make([]C.double, nrClasses, nrClasses)
	for i, cX := range cXs {
		C.predict_probability(model.cModel, cX, &proba[0])
		for j := 0; j < nrClasses; j++ {
			y.Set(i, j, float64(proba[j]))
		}
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
