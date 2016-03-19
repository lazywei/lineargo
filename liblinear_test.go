package liblinear_test

import (
	// "os"

	. "github.com/lazywei/liblinear"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("Liblinear", func() {
	Describe("Train and Predict", func() {
		It("should train and predict close result", func() {
			X, y := ReadLibsvm("test_fixture/heart_scale", true)
			model := Train(X, y, -1, L2R_LR, 1.0, 0.1, 0.01, map[int]float64{1: 1, -1: 1})
			y_pred := Predict(model, X)
			Expect(Accuracy(y, y_pred)).To(BeNumerically("~", 0.837037, 1e5))
		})
	})

	// Describe("SaveModel", func() {
	// 	It("should save a trained model", func() {
	// 		X, y := ReadLibsvm("test_fixture/heart_scale", true)
	// 		model := Train(X, y, -1, gParam)
	// 		filepath := "test_fixture/heart_scale.model.test"
	// 		SaveModel(model, filepath)
	// 		Expect(filepath).To(BeAnExistingFile())
	// 		os.Remove(filepath)
	// 	})
	// })

	Describe("LoadModel", func() {
		It("should panic if can not load from file", func() {
			Expect(func() { LoadModel("not_exist_model.model") }).To(Panic())
		})

		It("should load a trained model", func() {
			Expect(func() { LoadModel("test_fixture/heart_scale.model") }).NotTo(Panic())
		})
	})

	// Describe("PredictProba", func() {
	// 	It("should return probability estimation", func() {
	// 		X, y := ReadLibsvm("test_fixture/heart_scale", true)
	// 		model := Train(X, y, -1, gParam)
	// 		y_pred := PredictProba(model, X).RowView(0).RawVector().Data
	// 		Expect(y_pred[0]).To(BeNumerically("~", 0.95409, 1e-5))
	// 		Expect(y_pred[1]).To(BeNumerically("~", 0.0459103, 1e-5))
	// 	})
	// })
})
