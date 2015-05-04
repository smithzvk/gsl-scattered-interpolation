#include <config.h>
#include <assert.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_cblas.h>

#include "linear_simplex_util.h"
#include "linear_simplex.h"
#include "edge_flip.h"
#include "linear_simplex_integrity_check.h"

/* Test if the pair of simplexes are flippable (i.e. the result of a flip of
   that face to an edge would be a valid, non-reflex, simplicial complex).
   While we compute this, we fill in left_out, which holds the vertex indices
   that are left out of each potential new simplex */
static int
flippable(simplex_tree *tree, gsl_matrix *data,
          simplex_index leaf, int face,
          simplex_index neighbor, int far,
          int *left_out)
{
  int dim = tree->dim;
  int flippable = 1;

  gsl_vector_view p_face = DATA_POINT(data, POINT(leaf, face));
  gsl_vector_view p_far = DATA_POINT(data, POINT(neighbor, far));
  gsl_vector_view p_left_out;
  gsl_vector_view view;
  int ismplx;
  for (ismplx = 0; ismplx < dim; ismplx++)
    {
      int i;
      for (i = 0; i < dim + 1; i++)
        {
          /* "face" is the offset and left_out[ismplx] defines the normal */
          if (i == face) continue;
          int idx_on_face = i;
          if (idx_on_face > face) idx_on_face--;
          if (idx_on_face == ismplx)
            {
              left_out[ismplx] = i;
              continue;
            }
          if (idx_on_face > ismplx) idx_on_face--;

          /* This part only runs dim-1 times */
          gsl_vector_view p = DATA_POINT(data, POINT(leaf, i));
          view = gsl_matrix_row(tree->tmp_mat, idx_on_face);
          gsl_vector_memcpy(&(view.vector), &(p.vector));
          gsl_vector_sub(&(view.vector), &(p_face.vector));
        }
      /* Use the point left over to define the direction of the normal. */
      p_left_out = DATA_POINT(data, POINT(leaf, left_out[ismplx]));
      view = gsl_matrix_row(tree->tmp_mat, dim-1);
      gsl_vector_memcpy(&(view.vector), &(p_left_out.vector));
      gsl_vector_sub(&(view.vector), &(p_face.vector));

      orthogonalize(tree->tmp_mat);

      gsl_vector *v = tree->tmp_vec1;
      gsl_vector_memcpy(v, &(p_far.vector));
      gsl_vector_sub(v, &(p_face.vector));

      double proj;
      gsl_blas_ddot(&(view.vector), v, &proj);
      flippable &= (proj > 0);

      if (!flippable) break;
    }
  return flippable;
}

/* Compute and store the current external neighbors of a given simplex */
void static
save_current_neighbors(simplex_tree *tree,
                       simplex_index leaf, simplex_index neighbor,
                       simplex_index *old_neighbors)
{
  int k = 0;
  int ok = 0;
  int i;
  for (i = 0; i < tree->dim+1; i++)
    if (LINK(leaf, i) != neighbor)
      old_neighbors[k++] = LINK(leaf, i);
    else
      ok++;
  assert(("Inconsistency found in simplex tree structure, "
          "didn't find one and only one reverse link",
          ok == 1));
}

/* Set the points for a newly created simplex */
void static
set_points(simplex_tree *tree,
           simplex_index *new_simplexes, int ismplx,
           simplex_index leaf, int face,
           simplex_index neighbor, int far)
{
  POINT(new_simplexes[ismplx], 0) = POINT(leaf, face);
  POINT(new_simplexes[ismplx], 1) = POINT(neighbor, far);

  /* The points marked face and far in the original simplexes are the
     vertices that will correspond to all and only external faces in the
     new simplexes.  Face corresponds to the original first simplex and
     identifies external neighbors that are on the original second
     simplex, vice versa for far. */

  int j;
  for (j = 0; j < tree->dim + 1; j++)
    {
      /* We already included "face" */
      if (j == face) continue;
      int idx_on_face = j;
      if (idx_on_face > face) idx_on_face--;

      /* We need to exclude one point of the old simplex */
      if (idx_on_face == ismplx) continue;
      if (idx_on_face > ismplx) idx_on_face--;

      POINT(new_simplexes[ismplx], idx_on_face+2) = POINT(leaf, j);
    }
}

/* Set the external links of a new simplex */
void static
set_external_links(simplex_tree *tree, simplex_index *old_neighbors,
                   simplex_index neighbor, int neighbor_set,
                   simplex_index new_simplex,
                   int point_left_out)
{
  int dim = tree->dim;
  int jsmplx, null_option = 0;
  for (jsmplx = 0; jsmplx < dim; jsmplx++)
    {
      if (!old_neighbors[jsmplx])
        {
          null_option = 1;
          continue;
        }
      if (!point_in_simplex(tree, old_neighbors[jsmplx], point_left_out))
        break;
    }
  assert(("Inconsistency found in simplex tree structure, "
          "no proper external link found",
          jsmplx < dim || null_option));

  simplex_index ext;
  if (jsmplx < dim)
    ext = LINK(new_simplex, neighbor_set) = old_neighbors[jsmplx];
  else
    ext = LINK(new_simplex, neighbor_set) = 0;

  if (ext)
    {
      int j;
      FIND(j, LINK(ext, j) == neighbor, "no reverse link found");
      LINK(ext, j) = new_simplex;
    }
}

/* Set the internal links of a new simplex */
void static
set_internal_links(simplex_tree *tree,
                   simplex_index *new_simplexes, int ismplx)
{
  int i;
  int dim = tree->dim;
  for (i = 2; i < dim+1; i++)
    {
      int jsmplx;
      for (jsmplx = 0; jsmplx < dim; jsmplx++)
        {
          if (ismplx == jsmplx) continue;
          if (!point_in_simplex(tree, new_simplexes[jsmplx],
                                POINT(new_simplexes[ismplx], i)))
            break;
        }
      assert(("Inconsistency found in simplex tree structure, "
              "proper internal simplex not found",
              jsmplx < dim+1));
      LINK(new_simplexes[ismplx], i) = new_simplexes[jsmplx];
    }
}

/* Determine if the given pair of simplices can and should be "flipped" and, if
   so, perform the flip updating the simplex_tree data structure. */
int
delaunay(simplex_tree *tree, simplex_index leaf,
         gsl_matrix *data,
         /* face is the index in the simplex that identifies the face */
         int face,
         simplex_tree_accel *accel)
{
  int i;

  int dim = tree->dim;

  /* We never will flip the initial simplex */
  assert(leaf);
  if (!LINK(leaf, face)) return GSL_SUCCESS;
  assert(("Checking if a flip is necessary on an already flipped simplex",
          !SIMP(leaf)->flipped));
  assert(("Checking if a flip is necessary on a non-leaf simplex",
          SIMP(leaf)->leaf_p));

  simplex_index neighbor = LINK(leaf, face);
  /* The initial simplex should never be a neighbor */
  assert(neighbor);
  assert(("Inconsistency found in simplex tree structure, "
          "neighbor to leaf not a leaf",
          SIMP(neighbor)->leaf_p));

  /* Find far point */
  int far;
  FIND(far, LINK(neighbor, far) == leaf, "reverse link not found");

  int ret = 0;
  int *left_out = tree->left_out;
  if (flippable(tree, data, leaf, face, neighbor, far, left_out)
      && in_hypersphere(tree, leaf, data, POINT(neighbor, far), accel))
    {
      assert(("in_hypersphere not reciprocal",
              in_hypersphere(tree, neighbor, data, POINT(leaf, face), accel)));
      ret = 1;
      /* This should always be true as you will never flip an edge in 1D */
      assert(("Presumably trying to flip a 1D simplex?", dim > 1));

      /* We need to "flip" this face. */
      SIMP(leaf)->leaf_p = 0;
      SIMP(neighbor)->leaf_p = 0;
      SIMP(leaf)->flipped = 1;
      SIMP(neighbor)->flipped = 1;

      simplex_index *new_simplexes = tree->new_simplexes;
      simplex_index *old_neighbors1 = tree->old_neighbors1;
      simplex_index *old_neighbors2 = tree->old_neighbors2;

      /* Save current neighbors */
      save_current_neighbors(tree, leaf, neighbor, old_neighbors1);
      save_current_neighbors(tree, neighbor, leaf, old_neighbors2);

      /* Allocate new simplexes */
      int ismplx;
      for (ismplx = 0; ismplx < dim; ismplx++)
          new_simplexes[ismplx] = simplex_tree_node_alloc(tree);

      /* Set points on simplexes */
      for (ismplx = 0; ismplx < dim; ismplx++)
        set_points(tree, new_simplexes, ismplx, leaf, face, neighbor, far);

      /* External links */
      for (ismplx = 0; ismplx < dim; ismplx++)
        {
          set_external_links(tree, old_neighbors2, neighbor, 0,
                             new_simplexes[ismplx], POINT(leaf, left_out[ismplx]));
          set_external_links(tree, old_neighbors1, leaf, 1,
                             new_simplexes[ismplx], POINT(leaf, left_out[ismplx]));
        }

      /* Internal links */
      for (ismplx = 0; ismplx < dim; ismplx++)
        set_internal_links(tree, new_simplexes, ismplx);

      for (i = 0; i < dim; i++)
        {
          LINK(leaf, i) = new_simplexes[i];
          LINK(neighbor, i) = new_simplexes[i];
        }
      LINK(leaf, dim) = neighbor;
      LINK(neighbor, dim) = leaf;
      output_triangulation(tree, data, NULL,
                           "/tmp/grid_tri.dat", NULL, NULL);

      /* Now, recursively check the new external faces as they could now also
         need edge flips */
      for (ismplx = 0; ismplx < dim; ismplx++)
        for (i = 0; i < dim+1; i++)
          {
            if (SLINK(leaf, ismplx)->flipped) break;
            if (!LINK(LINK(leaf, ismplx), i)) continue;
            delaunay(tree, LINK(leaf, ismplx), data, i, accel);
            /* check_leaf_nodes(tree, NULL); */
            output_triangulation(tree, data, NULL,
                                 "/tmp/grid_tri.dat", NULL, NULL);
          }
    }

  return ret;
}
