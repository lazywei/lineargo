package lineargo

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/gonum/matrix/mat64"
)

func parseLibsvmElem(elem string) (int, float64) {
	elems := strings.Split(elem, ":")

	featureIdx, err := strconv.Atoi(elems[0])
	if err != nil {
		fmt.Println("Got error when parsing libsvm element index")
		panic(err)
	}

	featureVal, err := strconv.ParseFloat(elems[1], 64)
	if err != nil {
		fmt.Println("Got error when parsing libsvm element value")
		panic(err)
	}
	return featureIdx, featureVal
}

// ReadLibsvm reads libsvm format data from `filepath`. `oneBased` denotes the
// index of data stored in the file starts from 1 (`oneBased=true`) or 0
// (`oneBased=false`). Returned X, y is of dimension (nSamples, nFeatures) and
// (nSamples, 1) respectively.
func ReadLibsvm(filepath string, oneBased bool) (X, y *mat64.Dense) {
	type Data []string

	file, err := os.Open(filepath)
	if err != nil {
		fmt.Println("Got error when trying to open libsvm file")
		panic(err)
	}
	defer file.Close()

	nFeatures := 0
	nSamples := 0
	dataList := []Data{}

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		row := strings.Split(scanner.Text(), " ")
		dataList = append(dataList, row)

		if idx, _ := parseLibsvmElem(row[len(row)-1]); idx+1 > nFeatures {
			nFeatures = idx + 1
		}

		if oneBased {
			nFeatures = nFeatures - 1
		}

		nSamples++
	}

	X = mat64.NewDense(nSamples, nFeatures, nil)
	y = mat64.NewDense(nSamples, 1, nil)

	for i, data := range dataList {
		label, err := strconv.Atoi(data[0])
		if err != nil {
			fmt.Println("Got error when trying to set label for %v-th sample", i)
			panic(err)
		}
		y.Set(i, 0, float64(label))

		for k := 1; k < len(data); k++ {
			idx, val := parseLibsvmElem(data[k])

			if oneBased {
				X.Set(i, idx-1, float64(val))
			} else {
				X.Set(i, idx, float64(val))
			}

		}
	}

	if err := scanner.Err(); err != nil {
		fmt.Println("Got error when trying to read libsvm file")
		panic(err)
	}

	return
}
