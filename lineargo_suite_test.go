package lineargo_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"testing"
)

func TestLineargo(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Lineargo Suite")
}
