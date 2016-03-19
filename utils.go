package liblinear

/*
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"unsafe"
)

func mapCDouble(in []float64) []C.double {
	out := make([]C.double, len(in), len(in))
	for i, val := range in {
		out[i] = C.double(val)
	}
	return out
}

func mapCInt(in []int) []C.int {
	out := make([]C.int, len(in), len(in))
	for i, val := range in {
		out[i] = C.int(val)
	}
	return out
}

// convert C double pointer to float64 slice ...
func doubleToFloats(in *C.double, size int) []float64 {
	fmt.Println("in first")
	fmt.Println(*in)
	defer C.free(unsafe.Pointer(in))
	out := (*[1 << 30]float64)(unsafe.Pointer(in))[:size:size]
	return out
}
