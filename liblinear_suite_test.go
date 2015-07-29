package liblinear_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"testing"
)

func TestLiblinear(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Liblinear Suite")
}
