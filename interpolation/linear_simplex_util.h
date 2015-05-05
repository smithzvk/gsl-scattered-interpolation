#include <config.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_cblas.h>

#ifndef __GSL_LINEAR_SIMPLEX_UTIL_H__
#define __GSL_LINEAR_SIMPLEX_UTIL_H__

static inline int
singular (const gsl_matrix * LU)
{
  size_t i, n = LU->size1;

  for (i = 0; i < n; i++)
    {
      double u = gsl_matrix_get (LU, i, i);
      if (u == 0) return 1;
    }

  return 0;
}

static inline double
dnrm22(gsl_vector *v)
{
  int i;
  double mag2 = 0;
  for (i = 0; i < v->size; i++)
    {
      double val = gsl_vector_get(v, i);
      mag2 += val*val;
    }

  return mag2;
}

static inline int
orthogonalize(gsl_matrix *mat)
{
  int i;
  for (i = 0; i < mat->size1 - 1; i++)
    {
      gsl_vector_view vi = gsl_matrix_row(mat, i);
      int j;
      double vi_mag2 = dnrm22(&(vi.vector));
      for (j = i+1; j < mat->size1; j++)
        {
          gsl_vector_view vj = gsl_matrix_row(mat, j);
          double proj;
          gsl_blas_ddot(&(vi.vector), &(vj.vector), &proj);
          gsl_blas_daxpy(-proj/vi_mag2, &(vi.vector), &(vj.vector));

          double vj_mag2 = dnrm22(&(vj.vector));
          if (vj_mag2 < vi_mag2*10*GSL_DBL_EPSILON)
            /* The vectors don't span the space. */
            return GSL_FAILURE;
        }
    }

  return GSL_SUCCESS;
}

#endif
