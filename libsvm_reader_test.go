package lineargo_test

import (
	. "github.com/lazywei/lineargo"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("LibsvmReader", func() {
	Describe("ReadLibsvm", func() {

		It("should read libsvm format file", func() {
			X, y := ReadLibsvm("test_fixture/heart_scale", true)

			nSamples, _ := X.Dims()
			// Expect(nSamples).To(Equal(270))
			// Expect(nFeatures).To(Equal(13))

			nSamplesY, nColsY := y.Dims()
			Expect(nSamplesY).To(Equal(nSamples))
			Expect(nColsY).To(Equal(1))

			Expect(X.At(0, 0)).To(BeNumerically("==", 0.708333))
			Expect(X.At(0, 1)).To(BeNumerically("==", 1))
			Expect(X.At(0, 2)).To(BeNumerically("==", 1))
		})
	})
})
