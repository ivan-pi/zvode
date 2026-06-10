/* test_zvode_coupled.c
 * ============================================================
 *  C-native ZVODE test exercising the C API (extern/zvode.h),
 *  i.e. the bind(c) binding layer in extern/c_zvode.f90, including
 *  the user-supplied dense Jacobian path AND the user-data pointer
 *  (ctx), without going through the Python extension.
 *
 *  Problem : a coupled 2x2 complex linear system, dy/dt = A y, with
 *
 *              A = [ a   b ]      y(0) = [ 1 ]
 *                  [ b   a ]             [ 0 ]
 *
 *  The coefficients a and b are NOT hard-coded into the callbacks:
 *  they are carried in a `struct params` passed through ZVODE's opaque
 *  `ctx` pointer, so this also checks that ctx is forwarded intact to
 *  every F / JAC callback.  A broken ctx (NULL or wrong contents) yields
 *  the wrong A and trips the accuracy assertions below.
 *
 *  With a = -1, b = i, A is symmetric with eigenpairs
 *  (a+b, [1, 1]) and (a-b, [1, -1]), so for y(0) = [1, 0] the exact
 *  solution is
 *
 *              y1(t) = (e^{(a+b)t} + e^{(a-b)t}) / 2 = e^{-t} cos(t)
 *              y2(t) = (e^{(a+b)t} - e^{(a-b)t}) / 2 = i e^{-t} sin(t)
 *
 *  Method  : BDF (MF = 21), MITER = 1 -> user-supplied dense Jacobian.
 *  This drives c_fun_eval, c_jac_eval (Jacobian is the constant, fully
 *  dense matrix A), the dense LU in zvode_linalg (zgetrf / zgetrs), and
 *  ctx forwarding, all through the C entry point c_zvode.
 *
 *  Assertions (the returned int counts the failed assertion; 0 = pass):
 *    1. ISTATE == 2 on return
 *    2. T == TOUT on return
 *    3. |y1(TOUT) - exact| <= 1e-6
 *    4. |y2(TOUT) - exact| <= 1e-6
 * ============================================================
 */

#include <complex.h>
#include <math.h>
#include <stdio.h>

#include "zvode.h"

/* User data forwarded to every callback via ZVODE's ctx pointer. */
struct params {
    double complex a;   /* diagonal coefficient of A    */
    double complex b;   /* off-diagonal coefficient of A */
};

/* Right-hand side:  f = A y, with A = [[a, b], [b, a]] taken from ctx. */
static void fex(int neq, double t, const double complex y[],
                double complex ydot[], void *ctx)
{
    (void) neq; (void) t;
    /* A NULL ctx would mean the pointer was not forwarded; surface it as a
     * NaN derivative so the solver fails loudly instead of dereferencing it. */
    if (ctx == NULL) {
        ydot[0] = NAN;
        ydot[1] = NAN;
        return;
    }
    const struct params *p = (const struct params *) ctx;
    ydot[0] = p->a * y[0] + p->b * y[1];
    ydot[1] = p->b * y[0] + p->a * y[1];
}

/* Dense Jacobian:  pd[i + j*nrowpd] = df_i/dy_j  (column-major), = A. */
static void jex(int neq, double t, const double complex y[],
                int ml, int mu, double complex pd[], int nrowpd, void *ctx)
{
    (void) neq; (void) t; (void) y; (void) ml; (void) mu;
    if (ctx == NULL) {
        pd[0 + 0 * nrowpd] = NAN;
        pd[1 + 0 * nrowpd] = NAN;
        pd[0 + 1 * nrowpd] = NAN;
        pd[1 + 1 * nrowpd] = NAN;
        return;
    }
    const struct params *p = (const struct params *) ctx;
    pd[0 + 0 * nrowpd] = p->a;   /* df0/dy0 */
    pd[1 + 0 * nrowpd] = p->b;   /* df1/dy0 */
    pd[0 + 1 * nrowpd] = p->b;   /* df0/dy1 */
    pd[1 + 1 * nrowpd] = p->a;   /* df1/dy1 */
}

int main(void)
{
    /* Work-array minimums for MF = 21 (BDF, MITER = 1), NEQ = 2:
     *   LZW >= 8*NEQ + 2*NEQ**2 = 24,  LRW >= 20 + NEQ = 22,
     *   LIW >= 30 + NEQ = 32 (MITER != 0 needs the extra NEQ words). */
    enum { NEQ = 2, LZW = 24, LRW = 22, LIW = 32 };

    double complex y[NEQ], zwork[LZW];
    double rwork[LRW];
    int iwork[LIW];
    double t, tout;
    double rtol[1], atol[1];
    int itol, itask, istate, iopt, mf;

    double complex exact[NEQ];
    double err1, err2;
    const double tol = 1.0e-6;

    /* Model parameters, passed to the callbacks through ctx. */
    struct params p = { .a = -1.0, .b = I };

    /* Initial condition and solver settings */
    y[0]    = 1.0;
    y[1]    = 0.0;
    t       = 0.0;
    tout    = 3.0;
    itol    = 1;
    rtol[0] = 1.0e-8;
    atol[0] = 1.0e-10;
    itask   = 1;
    istate  = 1;
    iopt    = 0;
    mf      = 21;

    for (int i = 0; i < LZW; ++i) zwork[i] = 0.0;
    for (int i = 0; i < LRW; ++i) rwork[i] = 0.0;
    for (int i = 0; i < LIW; ++i) iwork[i] = 0;

    c_zvode(fex, NEQ, y, &t, tout, itol, rtol, atol, itask, &istate, iopt,
            zwork, LZW, rwork, LRW, iwork, LIW, jex, mf, &p);

    /* Assertion 1: successful return */
    if (istate != 2) {
        fprintf(stderr, "FAIL: c_zvode returned istate = %d\n", istate);
        return 1;
    }

    /* Assertion 2: reached TOUT */
    if (t < tout || t > tout) {
        fprintf(stderr, "FAIL: c_zvode did not reach TOUT, t = %.14e\n", t);
        return 2;
    }

    /* Assertions 3 & 4: solution accuracy.  The exact solution is derived
     * from the SAME parameters that were handed to the callbacks via ctx,
     * so this also confirms the callbacks saw the correct ctx contents. */
    exact[0] = 0.5 * (cexp((p.a + p.b) * tout) + cexp((p.a - p.b) * tout));
    exact[1] = 0.5 * (cexp((p.a + p.b) * tout) - cexp((p.a - p.b) * tout));
    err1 = cabs(y[0] - exact[0]);
    err2 = cabs(y[1] - exact[1]);

    printf("ZVODE coupled 2x2 C-API test  (dy/dt = A y, MF=21 user Jacobian, ctx)\n");
    printf("  y1 = %14.10f%+14.10fi   exact %14.10f%+14.10fi\n",
           creal(y[0]), cimag(y[0]), creal(exact[0]), cimag(exact[0]));
    printf("  y2 = %14.10f%+14.10fi   exact %14.10f%+14.10fi\n",
           creal(y[1]), cimag(y[1]), creal(exact[1]), cimag(exact[1]));
    printf("  |err1| = %10.3e   |err2| = %10.3e\n", err1, err2);
    printf("  NJE (Jacobian evaluations) = %d\n", iwork[12]);

    if (err1 > tol) {
        fprintf(stderr, "FAIL: |err1| = %10.3e > tol = %10.3e\n", err1, tol);
        return 3;
    }
    if (err2 > tol) {
        fprintf(stderr, "FAIL: |err2| = %10.3e > tol = %10.3e\n", err2, tol);
        return 4;
    }

    printf("PASS: test_zvode_coupled (C API, user Jacobian + ctx)\n");
    return 0;
}
