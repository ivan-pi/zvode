/* bdf.h -- C ABI for the Fortran BDF integrator (see c_bdf.f90).
 *
 * Lifecycle:  handle = bdf_create();  bdf_init(handle, ...);
 *             loop bdf_step / bdf_integrate;  read with bdf_get_*;
 *             bdf_destroy(handle).
 */
#ifndef BDF_H
#define BDF_H

#ifdef __cplusplus
extern "C" {
#endif

/* Jacobian sourcing (jac_mode). */
#define BDF_JAC_FD       0  /* internal finite differences          */
#define BDF_JAC_USER     1  /* user callback, depends on t and y    */
#define BDF_JAC_CONSTANT 2  /* user callback, evaluated once        */

/* Status / return codes (code out-arguments). */
#define BDF_OK              0
#define BDF_FINISHED        1
#define BDF_TOO_SMALL_STEP (-1)
#define BDF_TOO_MANY_STEPS (-2)

/* Right-hand side: write dy/dt at (t, y) into f[0..n-1]. */
typedef void (*bdf_rhs_fn)(
    int n, double t, const double *y, double *f, void *ctx);

/* Jacobian.  Dense (ml < 0): pd is column-major (n, n), pd[i + j*n] = df_i/dy_j.
 * Banded: pd is (ml+mu+1, n) in LAPACK band layout, pd[(i-j+mu) + j*ldpd]. */
typedef void (*bdf_jac_fn)(
    int n, double t, const double *y,
    int ml, int mu, double *pd, int ldpd, void *ctx);

/* Allocate an (uninitialised) solver handle.  Returns NULL on failure. */
void *bdf_create(void);

/* Configure / (re)initialise a handle.
 *   y0, atol   : length-n arrays (atol is per-component).
 *   jac        : may be NULL when jac_mode == BDF_JAC_FD.
 *   ml, mu     : band half-widths, or < 0 for a dense Jacobian.
 *   max_step   : <= 0 means "unbounded".
 *   first_step : <= 0 means "choose automatically".
 *   reuse_jac  : non-zero to cache/reuse the Jacobian between steps. */
void bdf_init(void *handle, int n, double t0, const double *y0, double t_bound,
              bdf_rhs_fn fun, double rtol, const double *atol, int jac_mode,
              bdf_jac_fn jac, int ml, int mu, double max_step,
              double first_step, int reuse_jac, void *ctx);

/* Move the integration boundary (direction unchanged). */
void bdf_set_t_bound(void *handle, double t_bound);

/* Take one internal step; *code receives a BDF_* status. */
void bdf_step(void *handle, int *code);

/* Step until the current t_bound is reached; *code receives a BDF_* status. */
void bdf_integrate(void *handle, int *code);

/* Accessors. */
double bdf_get_t(void *handle);
void   bdf_get_y(void *handle, int n, double *y);
void   bdf_get_stats(void *handle, int *nfev, int *njev, int *nlu, int *nsteps);

/* Release a handle created by bdf_create. */
void bdf_destroy(void *handle);

#ifdef __cplusplus
}
#endif

#endif /* BDF_H */
