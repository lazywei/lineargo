package liblinear

// #include "vendor/liblinear/linear.h"
import "C"
import "github.com/gonum/matrix/mat64"

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
