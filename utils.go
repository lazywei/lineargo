package liblinear

import "C"

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
