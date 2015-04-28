#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_cblas.h>

int
singular (const gsl_matrix * LU);

inline double
dnrm22(gsl_vector *v);

int
orthogonalize(gsl_matrix *mat);
