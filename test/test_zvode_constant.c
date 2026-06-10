/* test_zvode_constant.c
 * ============================================================
 *  C-native ZVODE test exercising the C API (extern/zvode.h),
 *  i.e. the bind(c) binding layer in extern/c_zvode.f90, without
 *  going through the Python extension.
 *
 *  Problem : dy/dt = i,  y(0) = i,  t in [0, 1]
 *  Exact   : y(t) = (1 + t) * i   =>   y(1) = 2 i
 *  Method  : Adams (MF = 10), no Jacobian (MITER = 0), NEQ = 1.
 *
 *  Mirrors the SciPy call sequence
 *      r = complex_ode(lambda t, x, arg: 1j)
 *      r.set_integrator('vode'); r.set_initial_value(1j); r.integrate(1)
 *
 *  Assertions (the returned int counts the failed assertion; 0 = pass):
 *    1. ISTATE == 2 on return
 *    2. |Re[y(1) - 2i]| + |Im[y(1) - 2i]| <= 1.0e-8
 * ============================================================
 */

#include <complex.h>
#include <math.h>
#include <stdio.h>

#include "zvode.h"

/* Right-hand side: dy/dt = i  (constant imaginary unit).
 * ctx is unused, matching the SciPy lambda that ignores its third argument. */
static void fex(int neq, double t, const double complex y[],
                double complex ydot[], void *ctx)
{
    (void) neq; (void) t; (void) y; (void) ctx;
    ydot[0] = I;
}

/* Jacobian stub: MF = 10 (MITER = 0) never calls the Jacobian.  It is passed
 * only to satisfy the call sequence and must never run. */
static void jex(int neq, double t, const double complex y[],
                int ml, int mu, double complex pd[], int nrowpd, void *ctx)
{
    (void) neq; (void) t; (void) y; (void) ml; (void) mu;
    (void) pd; (void) nrowpd; (void) ctx;
    fprintf(stderr, "BUG: Jacobian called for MF = 10 (MITER = 0)\n");
}

int main(void)
{
    /* Work-array minimums for MF = 10, NEQ = 1:
     *   LZW >= 15*NEQ = 15,  LRW >= 20 + NEQ = 21,  LIW >= 30. */
    enum { NEQ = 1, LZW = 15, LRW = 21, LIW = 30 };

    double complex y[NEQ], zwork[LZW];
    double rwork[LRW];
    int iwork[LIW];
    double t, tout;
    double rtol[1], atol[1];
    int itol, itask, istate, iopt, mf;

    double complex yexact, err;
    double aberr;
    const double tol = 1.0e-8;

    /* Initial condition and solver settings */
    y[0]    = I;            /* y(0) = i */
    t       = 0.0;
    tout    = 1.0;
    itol    = 1;
    rtol[0] = 1.0e-10;
    atol[0] = 1.0e-10;
    itask   = 1;
    istate  = 1;
    iopt    = 0;
    mf      = 10;

    for (int i = 0; i < LZW; ++i) zwork[i] = 0.0;
    for (int i = 0; i < LRW; ++i) rwork[i] = 0.0;
    for (int i = 0; i < LIW; ++i) iwork[i] = 0;

    c_zvode(fex, NEQ, y, &t, tout, itol, rtol, atol, itask, &istate, iopt,
            zwork, LZW, rwork, LRW, iwork, LIW, jex, mf, NULL);

    /* Assertion 1: successful return */
    if (istate != 2) {
        fprintf(stderr, "FAIL: c_zvode returned istate = %d\n", istate);
        return 1;
    }

    /* Assertion 2: solution accuracy */
    yexact = 2.0 * I;
    err    = y[0] - yexact;
    aberr  = fabs(creal(err)) + fabs(cimag(err));

    printf("ZVODE constant-RHS C-API test  (dy/dt = i, y(0) = i)\n");
    printf("  computed y(1) = %14.10f%+14.10fi\n", creal(y[0]), cimag(y[0]));
    printf("  exact    y(1) = %14.10f%+14.10fi\n", creal(yexact), cimag(yexact));
    printf("  |error|       = %10.3e\n", aberr);

    if (aberr > tol) {
        fprintf(stderr, "FAIL: |error| = %10.3e > tol = %10.3e\n", aberr, tol);
        return 2;
    }

    printf("PASS: test_zvode_constant (C API)\n");
    return 0;
}
