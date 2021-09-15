/**
 * \file:        cblas_aliases.h
 * \brief:       A file to allow passing float or doubles to cblas_d* functions
 * \author:      Hugh Delaney
 * \version:     
 * \date:        2021-09-16
 */
:
#ifndef CBLAS_ALIASES_3432
#define CBLAS_ALIASES_3432

void cublasDdot(cublasHandle_t handle, int n, float * x_d, int incx, float * y_d, int incy, float * ans_d) {
  cublasSdot(handle, n, x_d, incx, y_d, incy, ans_d);
}

void cublasDnrm2(cublasHandle_t handle, int n,float * x_d,int incx,float * ans_d){
  cublasSnrm2(handle,n,x_d,incx,ans_d);
}
float cblas_ddot(CBLAS_INDEX n, float * x, int incx, float * y, int incy) {
  return cblas_sdot(n, x, incx, y, incy);
}

float cblas_dnrm2(CBLAS_INDEX n, float * x, int incx) {
  return cblas_snrm2(n, x, incx);
}

void cblas_daxpy(CBLAS_INDEX n,float alpha, float * x, int incx, float* y, int incy) {
  cblas_saxpy(n, alpha, x, incx, y, incy);
}

void cblas_daxpby(CBLAS_INDEX n,float alpha, float * x, int incx,float beta, float* y, int incy) {
  cblas_saxpby(n, alpha, x, incx, beta, y, incy);
}
#endif
