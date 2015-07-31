package liblinear

/*
#cgo LDFLAGS: -llinear
#include <linear.h>
*/
import "C"

import (
	"github.com/davecgh/go-spew/spew"
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
		row := []*C.struct_feature_node{}
		for j := 0; j < nCols; j++ {
			val := X.At(i, j)
			if val != 0 {
				row = append(row, &C.struct_feature_node{
					index: C.int(j),
					value: C.double(val),
				})
			}
		}
		featureNodes = append(featureNodes, row[0])
	}

	return featureNodes
}

// struct model* train(const struct problem *prob, const struct parameter *param);
func MyTrain(X, y *mat64.Dense, bias bool, solverType int,
	eps, C_ float64,
	nrWeight int, weightLabel []int,
	weight []float64, p float64,
) *Model {
	nRows, nCols := X.Dims()

	var problem C.struct_problem
	cY := []C.double{}
	for _, v := range y.Col(nil, 0) {
		cY = append(cY, C.double(v))
	}
	problem.x = &(toFeatureNodes(X)[0])
	problem.y = &cY[0]
	problem.n = C.int(nRows)
	problem.l = C.int(nCols)

	var parameter C.struct_parameter
	parameter.solver_type = C.int(solverType)
	parameter.eps = C.double(eps)
	parameter.C = C.double(C_)
	parameter.nr_weight = C.int(nrWeight)
	// parameter.weight_label = C.int(&weightLabel[0])
	// parameter.weight= &weight[0]
	// parameter.p = C.double(p)

	model := C.train(&problem, &parameter)
	spew.Dump(model)
	return &Model{}
	// return &Model{
	// 	cModel: model,
	// }
}

// func main() {
// 	var problem C.struct_problem

// 	data := mat64.NewDense(2, 3, []float64{
// 		1, 3, 0,
// 		0, 2, 0,
// 	})

// 	y := []C.double{3, 2, 3}

// 	x := toFeatureNodes(data)

// 	problem.l = 3
// 	problem.n = 3
// 	problem.y = &y[0]
// 	problem.x = &x[0]
// }
