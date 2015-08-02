package liblinear_test

import (
	"os"

	. "github.com/lazywei/liblinear"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("Liblinear", func() {
	gParam := &Parameter{
		SolverType:   L2R_LR,
		C:            1.0,
		P:            0.1,
		Eps:          0.01,
		ClassWeights: map[int]float64{1: 1, -1: 1},
	}

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
			model := Train(X, y, -1, gParam)
			y_pred := Predict(model, X)
			Expect(Accuracy(y, y_pred)).To(BeNumerically("~", 0.837037, 1e5))
		})
	})

	Describe("SaveModel", func() {
		It("should save a trained model", func() {
			X, y := ReadLibsvm("test_fixture/heart_scale", true)
			model := Train(X, y, -1, gParam)
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

	Describe("PredictProba", func() {
		It("should return probability estimation", func() {
			X, y := ReadLibsvm("test_fixture/heart_scale", true)
			model := Train(X, y, -1, gParam)
			y_pred := PredictProba(model, X).Row(nil, 0)
			Expect(y_pred[0]).To(BeNumerically("~", 0.95409, 1e-5))
			Expect(y_pred[1]).To(BeNumerically("~", 0.0459103, 1e-5))
		})
	})
})
