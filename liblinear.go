package liblinear

/*
#cgo LDFLAGS: -llinear
#include <linear.h>
#include <stdio.h>

void myPrint(struct feature_node x) {
	printf("(%d, %f)\n", x.index, x.value);
}

struct model* goTrain(
	const struct problem *prob,
	const struct parameter *param) {

	printf("\n");
	printf("%f\n", prob->y[0]);
	printf("%f\n", prob->y[1]);
	printf("%f\n", prob->y[2]);
	printf("%f\n", prob->y[3]);
	printf("%f\n", prob->y[4]);
	printf("%f\n", prob->y[5]);

	return 0;
}


*/
import "C"

import "github.com/gonum/matrix/mat64"

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
	fns := toFeatureNodes(X)
	problem.x = &fns[0]
	problem.y = &cY[0]
	problem.n = C.int(nRows)
	problem.l = C.int(nCols)
	problem.bias = C.double(-1)

	var parameter C.struct_parameter
	parameter.solver_type = C.int(solverType)
	parameter.eps = C.double(eps)
	parameter.C = C.double(C_)
	parameter.nr_weight = C.int(nrWeight)
	parameter.weight_label = nil
	parameter.weight = nil
	parameter.p = C.double(p)

	model := C.train(&problem, &parameter)
	return &Model{
		cModel: model,
	}
}
