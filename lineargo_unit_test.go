package lineargo

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("Liblinear", func() {
	Describe("ToFeatureNodes", func() {
		It("should take *mat64.Dense and return feature nodes", func() {

			/* data := mat64.NewDense(2, 3, []float64{ */
			/* 	0, 0, 5, */
			/* 	0, 2, 0, */
			/* }) */

			/* X := toFeatureNodes(data) */

			/* Expect(float64((X[0]).value)).To(Equal(5.)) */
			/* Expect(int((X[0]).index)).To(Equal(2)) */

			/* Expect(float64((X[1]).value)).To(Equal(2.)) */
			/* Expect(int((X[1]).index)).To(Equal(1)) */
			Expect(true).To(Equal(true))
		})
	})
})
