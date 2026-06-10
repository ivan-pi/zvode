#ifndef ZVODE_H
#define ZVODE_H

#include <complex.h>

/* Right-hand side callback: writes the derivative into ydot[]. */
typedef void (*zvode_fun)(
    int neq,
    double t,
    const double complex y[],
    double complex ydot[],
    void *ctx);

/* Jacobian callback: writes df_i/dy_j into pd[] (column-major, ld = nrowpd). */
typedef void (*zvode_jac)(
    int neq,
    double t,
    const double complex y[],
    int ml, int mu,
    double complex pd[],
    int nrowpd,
    void *ctx);

/*
void zvode (
    void (*f) (),
    int neq,
    __GFORTRAN_DOUBLE_COMPLEX *y,
    double *t,
    double tout,
    int itol,
    const double *rtol,
    const double *atol,
    int itask,
    int *istate,
    int iopt,
    __GFORTRAN_DOUBLE_COMPLEX *zwork, int lzw,
    double *rwork, int lrw,
    int *iwork, int liw,
    void (*jac) (),
    int mf, void *ctx);
*/

void c_zvode(
    zvode_fun f,
    const int neq,
    double complex y[],
    double *t,
    const double tout,
    const int itol,
    const double rtol[],
    const double atol[],
    const int itask,
    int *istate,
    const int iopt,
    double complex zwork[], int lzw,
    double rwork[], int lrw,
    int iwork[], int liw,
    zvode_jac jac,
    const int mf,
    void *ctx);

struct zvode_step_t {
    double h, tn, hu;
    int nq;
};

int c_zvindy(
    int n,
    double t,
    const double complex yh[],
    int ldyh,
    int k,
    double complex dky[],
    const struct zvode_step_t *step);

#endif /* ZVODE_H */
