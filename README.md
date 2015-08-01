LIBLINEAR for Go
==========

This is a Golang wrapper for [LIBLINEAR (C.J. Lin et al)](http://ntucsu.csie.ntu.edu.tw/~cjlin/liblinear/) ([GitHub](https://github.com/cjlin1/liblinear)).
Note that the interface of this package might be slightly different from
liblinear C interface because of Go convention. Yet, I'll try to align the
function name and functionality to liblinear C library.

## Requirements

This package depends on LIBLINEAR 2.01+. Please install it via Homebrew or other
package managers on your OS:

```
brew update
brew info liblinear # make sure your formula will install version higher than 2.01
brew install liblinear
```

## Install

```
go get github.com/lazywei/liblinear
```

## Usage

Read libsvm format data into `*mat64.Dense` type. The recommended way to read
libsvm format file is through
[mockingbird](https://github.com/lazywei/mockingbird)

```go
# mockingbird.ReadLibsvm(filepath string, oneBased bool)
X, y := mockingbird.ReadLibsvm("heart_scale", true)
liblinear.Train(X, y, true, 0, 0.01, 1.0, 0, nil, nil, 0.1)
```
