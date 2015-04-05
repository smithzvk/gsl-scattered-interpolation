
#include <config.h>
#include <assert.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_interpsc.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_cblas.h>

/*

  For a thorough discussion of this method, see _Computational Geometry:
  Application and Algorithms_ chapter 9 or _Numerical Recipes: Third Edition_
  section 21.6.  The structure of the triangulation tree is as follows (I use the
  word triangle here, but all of this applies to arbitrary dimensionality):

  - This is not actually a tree but much about its use, structure, and properties
  are tree-like, so I call it one.

  - Leaf nodes correspond to triangles in the triangulation.  Internal nodes
  correspond to triangles that were part of the triangulation at some point in
  the building procedure, but have sense become invalid due to a sub-division
  and/or edge flip.

  - The internal nodes form a DAG usually moving from larger to smaller triangles
  which you can think of as sub-triangles, but this isn't strictly true due to
  edge flips.

  - The leaf nodes don't have sub-triangles, but they do have links to
  neighboring triangles, which is needed to create an efficient building
  algorithm.

  - The expected size of this tree will be 9N where N is the number of points, due
  to the randomness of the method, this is only expected.  Building the tree is
  expected to take O(N\log N) for a randomized data set (O(n^2) worst case).
  Finding a triangle that contains a point is expected to take O(\log N) time.

  A few other notes when interpreting this code.

  - All leaf nodes are linked to their neighbors except for the top-level
  triangle, which must be treated specially.  For this triangle, its neighbor
  links are NULL.

  - Faces, the things that separate simplexes/triangles, are defined by d unique
  points where d is the dimensionality.  In the context of a simplex, a face is
  identified by the index into the d+1 length array of data indexes that is not
  involved with the face.  This face index is the index into the links array.
  This is a nuanced structure.  Study the point insertion function to understand
  how this works.

*/

#include "linear_simplex.h"
#include "linear_simplex_integrity_check.h"

simplex_tree_node *
alloc_simplex_tree_node(int dim)
{
  int i;
  simplex_tree_node *tree = malloc(sizeof(simplex_tree));
  tree->points = malloc((dim+1) * sizeof(int));

  tree->leaf_p = 1;
  tree->n_links = dim+1;
  tree->links = malloc(tree->n_links * sizeof(simplex_tree_node *));
  return tree;
}

simplex_tree *
alloc_simplex_tree(int dim)
{
  int i, j;

  simplex_tree *tree = malloc(sizeof(simplex_tree));
  tree->dim = dim;
  tree->seed_points = gsl_matrix_alloc(dim+1, dim);
  /* Build a regular simplex, see:
     http://en.wikipedia.org/wiki/Simplex#Cartesian_coordinates_for_regular_n-dimensional_simplex_in_Rn */
  for (i = 0; i < dim; i++)
    {
      double tot2 = 0;
      for (j = 0; j < i; j++)
        {
          double comp = gsl_matrix_get(tree->seed_points, i, j);
          tot2 += comp*comp;
        }
      double chosen_component = sqrt(1 - tot2);
      gsl_matrix_set(tree->seed_points, i, i, chosen_component);

      double component_for_others = -(1.0/dim + tot2)/chosen_component;
      for (j = i+1; j < dim+1; j++)
        gsl_matrix_set(tree->seed_points, j, i, component_for_others);
    }
  /* Scale the simplex up to a large size.  This is necessary.  Why?  The
     algorithm is only correct when this is effectively infinity.  No matter how
     big, this could still cause problems.  Consider a nearly colinear set of
     points that defines a circle that will encompass a far vertex.  This will
     cause an edge flip that isn't correct.  However, we can perhaps fix this up
     in the in_hypersphere function (always return false for the far
     vertexes). */
  gsl_matrix_scale(tree->seed_points, 1e5);

  tree->n_points = 0;

  simplex_tree_node *node = alloc_simplex_tree_node(dim);
  for (i = 0; i < tree->dim+1; i++)
    {
      node->points[i] = -(i+1);
    }

  for (i = 0; i < node->n_links; i++)
    {
      /* This is a special triangle that doesn't have neighbors to consider */
      node->links[i] = NULL;
    }

  tree->root = node;

  return tree;
}

void
free_simplex_tree_node(simplex_tree_node *node)
{
  int i;
  if (!node->leaf_p)
    {
      for (i = 0; i < node->n_links; i++)
        {
          if (node->links[i])
            free_simplex_tree_node(node->links[i]);
        }
    }
  free(node->points);
  free(node->links);
  free(node);
}

void
free_simplex_tree(simplex_tree *tree)
{
  gsl_matrix_free(tree->seed_points);
  free_simplex_tree_node(tree->root);
  free(tree);
}

simplex_tree_accel *
alloc_simplex_tree_accel(size_t dim)
{
  simplex_tree_accel *accel = malloc(sizeof(simplex_tree_accel));
  accel->simplex_matrix = gsl_matrix_alloc(dim, dim);
  accel->perm = gsl_permutation_alloc(dim);
  accel->coords = gsl_vector_alloc(dim);
  accel->current_simplex = NULL;
}

void
free_simplex_tree_accel(simplex_tree_accel *accel)
{
  gsl_matrix_free(accel->simplex_matrix);
  gsl_permutation_free(accel->perm);
  gsl_vector_free(accel->coords);
  free(accel);
}

int
point_in_simplex(simplex_tree *tree, simplex_tree_node *node, int point)
{
  int dim = tree->dim;
  int i;
  for (i = 0; i < dim + 1; i++)
    {
      if (node->points[i] == point)
        break;
    }
  return (i < dim + 1);
}

simplex_tree_node *
find_leaf(simplex_tree *tree, gsl_matrix *data,
          gsl_vector *point,
          simplex_tree_accel *accel)
{
  simplex_tree_accel *local_accel = accel;
  int dim = tree->dim;
  if (!accel) local_accel = alloc_simplex_tree_accel(dim);

  simplex_tree_node *ret;
  if (contains_point(tree, tree->root, data, point, local_accel))
    ret = _find_leaf(tree, tree->root, data, point, local_accel);
  else
    /* Outside the cage.  I should handle this more gracefully.  Insuring that a
       point interpolation will return zero and that a point insertion will be
       meaningful would be enough. */
    assert(0);

  if (!accel) free_simplex_tree_accel(local_accel);

  return ret;
}

simplex_tree_node *
_find_leaf(simplex_tree *tree, simplex_tree_node *node, gsl_matrix *data,
           gsl_vector *point,
           simplex_tree_accel *accel)
{
  if (node->leaf_p)
    {
      return node;
    }
  else
    {
      int i;
      int dim = tree->dim;
      for (i = 0; i < node->n_links; i++)
        {
          if (node->links[i]
              && contains_point(tree, node->links[i], data, point, accel))
            {
              return _find_leaf(tree, node->links[i], data, point, accel);
            }
        }
    }
  /* We should never get here.  What if something falls through the cracks.
     This can happen if the point lies on a face.  Perhaps I could find the
     simplex that is the closest match and use that if none of them match.  This
     would be a useful fall-back. */
  assert(0);
}

int
insert_point(simplex_tree *tree, simplex_tree_node *leaf,
             gsl_matrix *data, gsl_vector *point,
             simplex_tree_accel *accel)
{
  int i, j;
  assert(accel);
  assert(leaf->leaf_p);
  leaf->leaf_p = 0;
  int dim = tree->dim;

  simplex_tree_node **new_simplexes = malloc((dim + 1) * sizeof(simplex_tree_node*));

  /* Allocate $d$ new simplexes */
  int ismplx;
  for (ismplx = 0; ismplx < dim + 1; ismplx++)
    {
      new_simplexes[ismplx] = alloc_simplex_tree_node(dim);
    }

  /* Populate the indexes in the newly created leaf nodes */
  for (i = 0; i < leaf->n_links; i++)
    {
      new_simplexes[i]->points[0] = tree->n_points;
      int k = 1;
      for (j = 0; j < dim+1; j++)
        {
          if (j==i) continue;
          new_simplexes[i]->points[k++] = leaf->points[j];
        }
    }

  /* Populate external links */
  for (i = 0; i < leaf->n_links; i++)
    {
      simplex_tree_node *neighbor = leaf->links[i];
      assert(!neighbor || neighbor->leaf_p);
      if (neighbor)
        assert(!point_in_simplex(tree, neighbor, leaf->points[i]));

      /* By convention, the 0-face is the one that already existed */
      new_simplexes[i]->links[0] = neighbor;

      /* Fix the neighbor links for the existing leaf nodes */
      if (neighbor)
        {
          for (j = 0; j < neighbor->n_links; j++)
            if (neighbor->links[j] == leaf)
              break;
          assert(j < neighbor->n_links);
          neighbor->links[j] = new_simplexes[i];
        }
    }

  /* Populate internal links */
  for (ismplx = 0; ismplx < dim+1; ismplx++)
    {
      int jsmplx;
      for (i = 1; i < dim+1; i++)
        {
          for (jsmplx = 0; jsmplx < dim+1; jsmplx++)
            {
              if (ismplx == jsmplx) continue;
              if (!point_in_simplex(tree, new_simplexes[jsmplx],
                                    new_simplexes[ismplx]->points[i]))
                break;
            }
          assert(jsmplx < dim+1);
          new_simplexes[ismplx]->links[i] = new_simplexes[jsmplx];
        }
    }

  for (i = 0; i < leaf->n_links; i++)
    leaf->links[i] = new_simplexes[i];
  free(new_simplexes);

  tree->n_points++;

  struct node_list *seen = NULL;
  check_leaf_nodes(tree, leaf->links[0], &seen);
  free_list(seen); seen = NULL;

  /* Now fix the Delaunay condition */
  for (i = 0; i < leaf->n_links; i++)
    {
      delaunay(tree, leaf->links[i], data, 0, accel);
      check_leaf_nodes(tree, leaf->links[0], &seen);
      free_list(seen); seen = NULL;
    }

  return GSL_SUCCESS;
}

/*

  A few things to note.  We are doing a generalized edge flip.  In 2D, this means
  removing two triangles and replacing them with two new triangles.  In $d$
  dimensions, this corresponds to removing 2 simplexes and replacing them with $d$
  simplexes.  The creation of the new simplexes is relatively simple, but due to
  the book-keeping involved in keeping track of the neighbors (some of which are
  newly created simplexes while others are external, previously existing
  simplexes), this is a bit complex.  This explains the high-level algorithm:

  For each new simplex $\{n f . idx\}$:

  External links:

  For each point, $i$, on the old separating face $F_n^{(n)} \equiv F_f^{(f)}$,
  define the edge that corresponds to removing $i$ from the face denoted $E_i$

  Link two simplexes to the new simplex, on face $F_n' \equiv \{f . E_i\}$ link
  the simplex across $F_i^{(f)}$.  On $F_f' \equiv \{n . E_i\}$ link the simplex
  across $F_i^{(n)}$.

  Internal links:

  It should be clear that new simplex should link with every other new simplex via
  some face.  This means that there must be $d-1$ internal links from each
  simplex.

  For each point, $i$, on the old separating face $F_n^{(n)}$, define the faces
  f_{ij} \equiv \{n f . idx_{-\{i,j\}}\} where $j$ is a index on edge $E_i$ (or,
  equivalently, $j \ne i$).  $f_{ij}$ should link to the new simplex $\{ n f i
  . idx_{-\{j\}} \}$.

  When building the new simplexes, you need only iterate over them with index
  $i$.  Then iterate over the edge with index $j$.  Then set links[0] and
  links[1] to the external simplexes, then set links[j+2] to...

*/

int
delaunay(simplex_tree *tree, simplex_tree_node *leaf,
         gsl_matrix *data,
         /* face is the index in the simplex that identifies the face */
         int face,
         simplex_tree_accel *accel)
{
  int i;

  int dim = tree->dim;

  if (!leaf->links[face]) return GSL_SUCCESS;
  assert(leaf->leaf_p);

  simplex_tree_node *neighbor = leaf->links[face];
  assert(neighbor->leaf_p);

  /* Find far point */
  int far;
  for (far = 0; far < neighbor->n_links; far++)
    {
      if (neighbor->links[far] == leaf)
        break;
    }
  assert(far < neighbor->n_links);

  if (in_hypersphere(tree, leaf, data, neighbor->points[far], accel))
    {
      /* This should always be true as you will never flip an edge in 1D */
      assert(dim > 1);

      /* We need to "flip" this face. */
      leaf->leaf_p = 0;
      neighbor->leaf_p = 0;
      simplex_tree_node **new_simplexes = malloc(dim * sizeof(simplex_tree_node*));

      /* Save current neighbors */
      simplex_tree_node **old_neighbors1 = malloc(dim * sizeof(simplex_tree_node*));
      simplex_tree_node **old_neighbors2 = malloc(dim * sizeof(simplex_tree_node*));
      int *left_out = malloc(dim * sizeof(int));

      {
        int k = 0;
        int ok = 0;
        for (i = 0; i < leaf->n_links; i++)
          if (leaf->links[i] != neighbor)
            old_neighbors1[k++] = leaf->links[i];
          else
            ok = 1;
        assert(ok);
        ok = 0;
        k = 0;
        for (i = 0; i < neighbor->n_links; i++)
          if (neighbor->links[i] != leaf)
            old_neighbors2[k++] = neighbor->links[i];
          else
            ok = 1;
        assert(ok);
      }

      /* Allocate new simplexes */
      int ismplx;
      for (ismplx = 0; ismplx < dim; ismplx++)
        {
          new_simplexes[ismplx] = alloc_simplex_tree_node(dim);
        }

      /* Set points on simplexes */
      for (ismplx = 0; ismplx < dim; ismplx++)
        {
          new_simplexes[ismplx]->points[0] = leaf->points[face];
          new_simplexes[ismplx]->points[1] = neighbor->points[far];

          /* The points marked face and far in the original simplexes are the
             vertices that will correspond to all and only external faces in the
             new simplexes.  Face corresponds to the original first simplex and
             identifies external neighbors that are on the original second
             simplex, vice versa for far. */

          int j;
          for (j = 0; j < dim + 1; j++)
            {
              /* We already included "face" */
              if (j == face) continue;
              int idx_on_face = j;
              if (idx_on_face > face) idx_on_face--;

              /* We need to exclude one point of the old simplex */
              if (idx_on_face == ismplx)
                {
                  left_out[ismplx] = j;
                  continue;
                }
              if (idx_on_face > ismplx) idx_on_face--;

              new_simplexes[ismplx]->points[idx_on_face+2] = leaf->points[j];
            }
        }

      /* External links */
      for (ismplx = 0; ismplx < dim; ismplx++)
        {
          int jsmplx;
          int null_option = 0;
          for (jsmplx = 0; jsmplx < dim; jsmplx++)
            {
              if (!old_neighbors2[jsmplx])
                {
                  null_option = 1;
                  continue;
                }
              if (!point_in_simplex(tree, old_neighbors2[jsmplx],
                                    leaf->points[left_out[ismplx]]))
                break;
            }
          assert(jsmplx < dim || null_option);

          {
            simplex_tree_node *ext;
            if (jsmplx < dim)
              ext = new_simplexes[ismplx]->links[0] = old_neighbors2[jsmplx];
            else
              ext = new_simplexes[ismplx]->links[0] = NULL;

            if (ext)
              {
                int j;
                for (j = 0; j < ext->n_links; j++)
                  if (ext->links[j] == neighbor)
                    break;
                assert(j < ext->n_links);
                ext->links[j] = new_simplexes[ismplx];
              }
          }


          null_option = 0;
          for (jsmplx = 0; jsmplx < dim; jsmplx++)
            {
              if (!old_neighbors1[jsmplx])
                {
                  null_option = 1;
                  continue;
                }
              if (!point_in_simplex(tree, old_neighbors1[jsmplx],
                                    leaf->points[left_out[ismplx]]))
                break;
            }
          assert(jsmplx < dim || null_option);

          {
            simplex_tree_node *ext;
            if (jsmplx < dim)
              ext = new_simplexes[ismplx]->links[1] = old_neighbors1[jsmplx];
            else
              ext = new_simplexes[ismplx]->links[1] = NULL;

            if (ext)
              {
                int j;
                for (j = 0; j < ext->n_links; j++)
                  if (ext->links[j] == leaf)
                    break;
                assert(j < ext->n_links);
                ext->links[j] = new_simplexes[ismplx];
              }
          }
        }

      /* Internal links */
      for (ismplx = 0; ismplx < dim; ismplx++)
        {
          int jsmplx;
          for (i = 2; i < dim+1; i++)
            {
              for (jsmplx = 0; jsmplx < dim; jsmplx++)
                {
                  if (ismplx == jsmplx) continue;
                  if (!point_in_simplex(tree, new_simplexes[jsmplx],
                                        new_simplexes[ismplx]->points[i]))
                    break;
                }
              assert(jsmplx < dim+1);
              new_simplexes[ismplx]->links[i] = new_simplexes[jsmplx];
            }
        }

      leaf->n_links = dim;
      neighbor->n_links = dim;
      for (i = 0; i < leaf->n_links; i++)
        {
          leaf->links[i] = new_simplexes[i];
          neighbor->links[i] = new_simplexes[i];
        }
      free(new_simplexes);
      free(old_neighbors1);
      free(old_neighbors2);
      free(left_out);

      /* Now, recursively check the new external faces as they could now also
         need edge flips */
      for (ismplx = 0; ismplx < dim; ismplx++)
        for (i = 0; i < dim+1; i++)
          {
            if (!leaf->links[ismplx]->links[i]) continue;
            int ret = delaunay(tree, leaf->links[ismplx]->links[i], data, i, accel);
            if (GSL_SUCCESS != ret)
              return ret;
          }
    }

  return GSL_SUCCESS;
}

inline double
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

int
in_hypersphere(simplex_tree *tree, simplex_tree_node *node,
               gsl_matrix *data,
               int idx, simplex_tree_accel *accel)
{
  /* /\* Ensure that all of the seed_points are always outside of any */
  /*    hypersphere. *\/ */
  /* if (idx < 0) */
  /*   return 0; */

  gsl_vector *x0 = gsl_vector_alloc(tree->dim);
  double r2;

  gsl_vector_view point;
  if (idx < 0)
    point = gsl_matrix_row(tree->seed_points, -idx - 1);
  else
    point = gsl_matrix_row(data, idx);

  calculate_hypersphere(tree, node, data, x0, &r2, accel);

  /* Compute the displacement vector */
  gsl_vector_sub(x0, &(point.vector));

  double dist2 = dnrm22(x0);

  gsl_vector_free(x0);
  return dist2 < r2;
}

/* See http://steve.hollasch.net/cgindex/geometry/sphere4pts.html (in
   particular, Dr. John S. Eickemeyer's suggestion) on how to do this in a way
   that generalizes to arbitrary dimensionality. */
int
calculate_hypersphere(simplex_tree *tree, simplex_tree_node *node,
                      gsl_matrix *data,
                      gsl_vector *x0, double *r2,
                      simplex_tree_accel *accel)
{
  int i, j;
  int dim = tree->dim;
  accel->current_simplex = NULL;

  for (i = 0; i < dim; i++)
    {
      gsl_vector_set(accel->coords, i, 0);

      gsl_vector_view vi, vi1;
      int vi_idx = node->points[i];
      int vi1_idx = node->points[i+1];

      if (vi_idx < 0)
        vi = gsl_matrix_row(tree->seed_points, -vi_idx - 1);
      else
        vi = gsl_matrix_row(data, vi_idx);

      if (vi1_idx < 0)
        vi1 = gsl_matrix_row(tree->seed_points, -vi1_idx - 1);
      else
        vi1 = gsl_matrix_row(data, vi1_idx);

      for (j = 0; j < dim; j++)
        {
          double pij = gsl_vector_get(&(vi.vector), j);
          double pij1 = gsl_vector_get(&(vi1.vector), j);
          gsl_vector_set(accel->coords, i, (gsl_vector_get(accel->coords, i)
                                            + pij*pij - pij1*pij1));
          gsl_matrix_set(accel->simplex_matrix, i, j, pij - pij1);
        }
      gsl_vector_set(accel->coords, i, 0.5 * gsl_vector_get(accel->coords, i));
    }

  int signum;
  gsl_linalg_LU_decomp(accel->simplex_matrix, accel->perm, &signum);
  gsl_linalg_LU_solve(accel->simplex_matrix, accel->perm,
                      accel->coords, x0);

  gsl_vector_view first_point;
  int idx = node->points[0];
  if (idx < 0)
    first_point = gsl_matrix_row(tree->seed_points, -idx - 1);
  else
    first_point = gsl_matrix_row(data, idx);

  gsl_vector_memcpy(accel->coords, &(first_point.vector));
  /* Why doesn't this work? */
  /* gsl_blas_daxpy(-1, x0, accel->coords); */
  gsl_vector_sub(accel->coords, x0);
  *r2 = dnrm22(accel->coords);

  return GSL_SUCCESS;
}

int
calculate_bary_coords(simplex_tree *tree, simplex_tree_node *node, gsl_matrix *data,
                      gsl_vector *point,
                      simplex_tree_accel *accel)
{
  assert(accel != NULL);
  int dim = tree->dim;
  gsl_vector_view x0;
  int x0_idx= node->points[dim];

  if (x0_idx < 0)
    x0 = gsl_matrix_row(tree->seed_points, -x0_idx - 1);
  else
    x0 = gsl_matrix_row(data, x0_idx);

  if (node != accel->current_simplex)
    {
      accel->current_simplex = node;

      int i, j;

      for (i = 0; i < dim; i++)
        {
          gsl_vector_view p;
          int xi_idx = node->points[i];
          if (xi_idx < 0)
            p = gsl_matrix_row(tree->seed_points, -xi_idx - 1);
          else
            p = gsl_matrix_row(data, xi_idx);
          for (j = 0; j < dim; j++)
            {
              gsl_matrix_set(accel->simplex_matrix, j, i,
                             gsl_vector_get(&(p.vector), j)
                             - gsl_vector_get(&(x0.vector), j));
            }
        }

      int signum;
      gsl_linalg_LU_decomp(accel->simplex_matrix, accel->perm, &signum);
    }

  gsl_vector *pp = gsl_vector_alloc(dim);
  gsl_vector_memcpy(pp, point);
  gsl_vector_sub(pp, &(x0.vector));
  gsl_linalg_LU_solve(accel->simplex_matrix, accel->perm, pp, accel->coords);
  gsl_vector_free(pp);
  return GSL_SUCCESS;
}

int
contains_point(simplex_tree *tree, simplex_tree_node *node,
               gsl_matrix *data, gsl_vector *point,
               simplex_tree_accel *accel)
{
  int i;
  int dim = tree->dim;

  calculate_bary_coords(tree, node, data, point, accel);

  double tot = 0;
  for (i = 0; i < dim; i++)
    {
      double coord = gsl_vector_get(accel->coords, i);
      tot += coord;
      if ((coord < 0) || (coord > 1))
        return 0;
    }
  if ((tot < 0) || (tot > 1))
    return 0;
  return 1;
}

double
interp_point(simplex_tree *tree, simplex_tree_node *leaf,
             gsl_matrix *data, gsl_vector *response, gsl_vector *point,
             simplex_tree_accel *accel)
{
  int i;
  int dim = tree->dim;

  assert(leaf->leaf_p);
  calculate_bary_coords(tree, leaf, data, point, accel);

  double tot = 0;
  double interp = 0;
  for (i = 0; i < dim; i++)
    {
      double coord = gsl_vector_get(accel->coords, i);
      tot += coord;
      int xi_idx = leaf->points[i];
      /* Only add a contribution if the point isn't a seed point */
      if (xi_idx >= 0)
        interp += coord * gsl_vector_get(response, xi_idx);
    }
  int idx = leaf->points[dim];
  /* Only add a contribution if the point isn't a seed point */
  if (idx >= 0)
    interp += (1 - tot) * gsl_vector_get(response, idx);
  return interp;
}
