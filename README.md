LIBLINEAR for Go
==========

This is a Golang wrapper for [LIBLINEAR (C.J. Lin et al)](http://ntucsu.csie.ntu.edu.tw/~cjlin/liblinear/) ([GitHub](https://github.com/cjlin1/liblinear)).

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
