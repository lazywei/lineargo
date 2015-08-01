package liblinear_test

import (
	. "github.com/lazywei/liblinear"
	"github.com/lazywei/mockingbird"

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
			X, y := mockingbird.ReadLibsvm("heart_scale", true)
			model := Train(X, y, -1, 0, 0.01, 1.0, 0, nil, nil, 0.1)
			y_pred := Predict(model, X)
			Expect(Accuracy(y, y_pred)).To(BeNumerically("~", 0.837037, 1e5))
		})
	})
})
