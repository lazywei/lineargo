package lineargo

/*
#cgo LDFLAGS: -llinear
#include <linear.h>
#include <stdio.h>
*/
import "C"

const (
	// Solver Type
	L2R_LR              = int(C.L2R_LR)
	L2R_L2LOSS_SVC_DUAL = int(C.L2R_L2LOSS_SVC_DUAL)
	L2R_L2LOSS_SVC      = int(C.L2R_L2LOSS_SVC)
	L2R_L1LOSS_SVC_DUAL = int(C.L2R_L1LOSS_SVC_DUAL)
	MCSVM_CS            = int(C.MCSVM_CS)
	L1R_L2LOSS_SVC      = int(C.L1R_L2LOSS_SVC)
	L1R_LR              = int(C.L1R_LR)
	L2R_LR_DUAL         = int(C.L2R_LR_DUAL)
	L2R_L2LOSS_SVR      = int(C.L2R_L2LOSS_SVR)
	L2R_L2LOSS_SVR_DUAL = int(C.L2R_L2LOSS_SVR_DUAL)
	L2R_L1LOSS_SVR_DUAL = int(C.L2R_L1LOSS_SVR_DUAL)
)
