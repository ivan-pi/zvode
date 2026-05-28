#include <complex.h>

void (*zvode_fun)(
    int neq,
    double t,
    double complex y[],
    double complex ydot[],
    void *ctx);

void (*zvode_jac)(
    int neq
    double t
    double complex y[],
    int ml, int mu,
    double complex pd[],
    int nrowpd,
    void *ctx);

void zvode(
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

void zvindy(
    double t,
    int k,
    double complex yh[],
    int neq,
    double complex dky[],
    int *iflag);
