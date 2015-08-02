package liblinear_test

import (
	"os"

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

	Describe("Train and Predict", func() {
		It("should return a feature node", func() {
			X, y := ReadLibsvm("test_fixture/heart_scale", true)
			model := Train(X, y, -1, 0, 0.01, 1.0, 0, nil, nil, 0.1)
			y_pred := Predict(model, X)
			Expect(Accuracy(y, y_pred)).To(BeNumerically("~", 0.837037, 1e5))
		})
	})

	Describe("SaveModel", func() {
		It("should save a trained model", func() {
			X, y := ReadLibsvm("test_fixture/heart_scale", true)
			model := Train(X, y, -1, 0, 0.01, 1.0, 0, nil, nil, 0.1)
			filepath := "test_fixture/heart_scale.model.test"
			SaveModel(model, filepath)
			Expect(filepath).To(BeAnExistingFile())
			os.Remove(filepath)
		})
	})

	Describe("LoadModel", func() {
		It("should panic if can not load from file", func() {
			Expect(func() { LoadModel("not_exist_model.model") }).To(Panic())
		})

		It("should load a trained model", func() {
			Expect(func() { LoadModel("test_fixture/heart_scale.model") }).NotTo(Panic())
		})
	})
})
