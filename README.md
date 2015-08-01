LIBLINEAR for Go
==========

This is a Golang wrapper for [LIBLINEAR (C.J. Lin et al)](http://ntucsu.csie.ntu.edu.tw/~cjlin/liblinear/) ([GitHub](https://github.com/cjlin1/liblinear)).
Note that the interface of this package might be slightly different from
liblinear C interface because of Go convention. Yet, I'll try to align the
function name and functionality to liblinear C library.

## Introduction to LIBLINEAR

LIBLINEAR is a linear classifier for data with millions of instances and features. It supports

- L2-regularized classifiers
- L2-loss linear SVM, L1-loss linear SVM, and logistic regression (LR)
- L1-regularized classifiers (after version 1.4)
- L2-loss linear SVM and logistic regression (LR)
- L2-regularized support vector regression (after version 1.9)
- L2-loss linear SVR and L1-loss linear SVR.


## Install

This package depends on LIBLINEAR 2.01+. Please install it first via Homebrew or
other package managers on your OS:

```
brew update
brew info liblinear # make sure your formula will install version higher than 2.01
brew install liblinear
```

After liblinear installation, just `go get` this package

```
go get github.com/lazywei/liblinear
```

## Usage

Read libsvm format data into `*mat64.Dense` type. The recommended way to read
libsvm format file is through
[mockingbird](https://github.com/lazywei/mockingbird):

```go
import linear "github.com/lazywei/liblinear"

// mockingbird.ReadLibsvm(filepath string, oneBased bool) (X, y *mat64.Dense)
X, y := mockingbird.ReadLibsvm("heart_scale", true)

// linear.Train(
// X, y *mat64.Dense,
// bias float64,
// solverType int,
// eps, C_ float64,
// nrWeight int, weightLabel []int,
// weight []float64, p float64) (*Model)
// Please checkout liblinear's doc for the explanation for these parameters.
model := linear.Train(X, y, true, 0, 0.01, 1.0, 0, nil, nil, 0.1)
y_pred:= linear.Predict(model, X)

fmt.Println(linear.Accuracy(y, y_pred))
```

## Roadmap

- [ ] Wrap core functions
  - [x] Train
  - [x] Predict
  - [ ] Predict Probability
  - [ ] Cross Validation
  - [ ] Save Model / Load Model
- [ ] Better Wrapper for liblinear's C struct
  - [x] Model (wrap `struct model`)
  - [x] FeatureNode (wrap `struct feature_node`)
  - [ ] Parameter (wrap `struct parameter`)
  - [ ] Problem (wrap `struct problem`)
- [ ] Abstraction for classifier: encapsulate `model`
