package liblinear_test

import (
	"github.com/gonum/matrix/mat64"
	. "github.com/lazywei/liblinear"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("Liblinear", func() {
	Describe("NewFeatureNode", func() {
		It("should return a feature node", func() {
			fn := NewFeatureNode(3, 5.0)
			index, value := fn.GetData()
			Expect(index).To(Equal(3))
			Expect(value).To(Equal(5.0))
		})
	})

	Describe("Train", func() {
		It("should return a feature node", func() {
			X := mat64.NewDense(2, 3, []float64{
				0, 1, 3,
				2, 0, 5,
			})
			y := mat64.NewDense(2, 1, []float64{
				0, 1,
			})
			MyTrain(X, y, true, 0, 1e-5, 1.0, 1, nil, nil, 1)
		})
	})
})
