package lineargo

/*
#include <stdlib.h>
*/
import "C"
import "unsafe"

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
	defer C.free(unsafe.Pointer(in))
	outD := (*[1 << 30]C.double)(unsafe.Pointer(in))[:size:size]
	out := make([]float64, size, size)
	for i := 0; i < size; i++ {
		out[i] = float64(outD[i])
	}

	return out
}
