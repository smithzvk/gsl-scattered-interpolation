#include <config.h>
#include <assert.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>

#include "linear_simplex.h"
#include "linear_simplex_util.h"

#ifndef __GSL_EDGE_FLIP_H__
#define __GSL_EDGE_FLIP_H__

static int
flippable(simplex_tree *tree, gsl_matrix *data,
          simplex_index leaf, int face,
          simplex_index neighbor, int far,
          int *left_out);

static void
set_left_out(simplex_tree *tree, int face, int *left_out);

static void
save_current_neighbors(simplex_tree *tree,
                       simplex_index leaf, simplex_index neighbor,
                       simplex_index *old_neighbors);

static void
set_points(simplex_tree *tree,
           simplex_index *new_simplexes, int ismplx,
           simplex_index leaf, int face,
           simplex_index neighbor, int far);

static void
set_external_links(simplex_tree *tree, simplex_index *old_neighbors,
                   simplex_index neighbor, int neighbor_set,
                   simplex_index new_simplex,
                   int point_left_out);

static void
set_internal_links(simplex_tree *tree,
                   simplex_index *new_simplexes, int ismplx);

int
delaunay(simplex_tree *tree, simplex_index leaf,
         gsl_matrix *data,
         /* face is the index in the simplex that identifies the face */
         int face,
         simplex_tree_accel *accel);

#endif
