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

void static
save_current_neighbors(simplex_tree *tree,
                       simplex_index leaf, simplex_index neighbor,
                       simplex_index *old_neighbors);

void static
set_points(simplex_tree *tree,
           simplex_index *new_simplexes, int ismplx,
           simplex_index leaf, int face,
           simplex_index neighbor, int far);

void static
set_external_links(simplex_tree *tree, simplex_index *old_neighbors,
                   simplex_index neighbor, int neighbor_set,
                   simplex_index new_simplex,
                   int point_left_out);

void static
set_internal_links(simplex_tree *tree,
                   simplex_index *new_simplexes, int ismplx);

int
delaunay(simplex_tree *tree, simplex_index leaf,
         gsl_matrix *data,
         /* face is the index in the simplex that identifies the face */
         int face,
         simplex_tree_accel *accel);

#endif
