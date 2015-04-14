
#include <config.h>
#include <assert.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
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
simplex_tree_node_alloc(int dim)
{
  int i;
  simplex_tree_node *tree = malloc(sizeof(simplex_tree));
  tree->points = malloc((dim+1) * sizeof(int));

  tree->leaf_p = 1;
  tree->flipped = 0;
  tree->n_links = dim+1;
  tree->links = malloc(tree->n_links * sizeof(simplex_tree_node *));
  return tree;
}

void
simplex_tree_node_free(simplex_tree *tree, simplex_tree_node *node)
{
  int i;
  int dim = tree->dim;
  if (!node->leaf_p && !(node->flipped && (node < node->links[dim])))
    {
      for (i = 0; i < node->n_links; i++)
        {
          if (node->links[i])
            simplex_tree_node_free(tree, node->links[i]);
        }
    }
  free(node->points);
  free(node->links);
  free(node);
}

simplex_tree *
simplex_tree_alloc(int dim, int n_points)
{
  int i, j;

  simplex_tree *tree = malloc(sizeof(simplex_tree));
  tree->dim = dim;
  tree->seed_points = gsl_matrix_alloc(dim+1, dim);
  tree->n_points = 0;

  simplex_tree_node *node = simplex_tree_node_alloc(dim);
  tree->root = node;

  tree->accel = simplex_tree_accel_alloc(dim);
  tree->new_simplexes = malloc((dim+1) * sizeof(simplex_tree_node*));
  tree->old_neighbors1 = malloc(dim * sizeof(simplex_tree_node*));
  tree->old_neighbors2 = malloc(dim * sizeof(simplex_tree_node*));
  tree->left_out = malloc(dim * sizeof(int));
  tree->shift = gsl_vector_alloc(dim);
  tree->scale = gsl_vector_alloc(dim);
  tree->min = gsl_vector_alloc(dim);
  tree->max = gsl_vector_alloc(dim);
  tree->shuffle = gsl_permutation_alloc(n_points);

  return tree;
}

void
simplex_tree_free(simplex_tree *tree)
{
  gsl_matrix_free(tree->seed_points);
  simplex_tree_node_free(tree, tree->root);
  simplex_tree_accel_free(tree->accel);
  free(tree->new_simplexes);
  free(tree->old_neighbors1);
  free(tree->old_neighbors2);
  free(tree->left_out);
  free(tree->shift);
  free(tree->scale);
  free(tree->min);
  free(tree->max);
  gsl_permutation_free(tree->shuffle);

  free(tree);
}

int
simplex_tree_init(simplex_tree *tree, gsl_matrix *data,
                  gsl_vector *min, gsl_vector *max,
                  int init_flags, gsl_rng *rng)
{
  int i, dim = tree->dim;

  assert(data || (min && max) || (init_flags & SIMPLEX_TREE_NOSTANDARDIZE));
  if (init_flags & SIMPLEX_TREE_NOSTANDARDIZE)
    {
      for (i = 0; i < dim; i++)
        {
          gsl_vector_set(tree->shift, i, 0);
          gsl_vector_set(tree->scale, i, 1);
        }
    }
  else if (min && max)
    {
      for (i = 0; i < dim; i++)
        {
          double min_val = gsl_vector_get(min, i);
          double max_val = gsl_vector_get(max, i);
          gsl_vector_set(tree->shift, i, (min_val + max_val)/2.0);
          if (max_val - min_val <= 0)
            /* Something is wrong, but last ditch effort to make this work */
            gsl_vector_set(tree->scale, i, 1.0);
          else
            gsl_vector_set(tree->scale, i, 1.0/(max_val - min_val));
        }
    }
  else if (data && (!min || !max))
    {
      for (i = 0; i < dim; i++)
        {
          if (min)
            gsl_vector_set(tree->min, i, gsl_vector_get(min, i));
          else
            gsl_vector_set(tree->min, i, gsl_matrix_get(data, 0, i));

          if (max)
            gsl_vector_set(tree->max, i, gsl_vector_get(max, i));
          else
            gsl_vector_set(tree->max, i, gsl_matrix_get(data, 0, i));
        }

      for (i = 1; i < data->size1; i++)
        {
          int j;
          for (j = 0; j < dim; j++)
            {
              double val = gsl_matrix_get(data, i, j);
              if (!min && val < gsl_vector_get(tree->min, j))
                gsl_vector_set(tree->min, j, val);
              if (!max && val > gsl_vector_get(tree->max, j))
                gsl_vector_set(tree->max, j, val);
            }
        }
      for (i = 0; i < dim; i++)
        {
          double min_val = gsl_vector_get(tree->min, i);
          double max_val = gsl_vector_get(tree->max, i);
          gsl_vector_set(tree->shift, i, (min_val + max_val)/2.0);
          if (max_val - min_val <= 0)
            /* Something is wrong, but last ditch effort to make this work */
            gsl_vector_set(tree->scale, i, 1.0);
          else
            gsl_vector_set(tree->scale, i, 1.0/(max_val - min_val));
        }
    }
  else
    {
      /* You need to either provide a min and max, or provide some
         representative data to infer one from, or tell the algorithm to not
         standardize with SIMPLEX_TREE_NOSTANDARDIZE. */
      return GSL_FAILURE;
    }

  /* Build a regular simplex, see:
     http://en.wikipedia.org/wiki/Simplex#Cartesian_coordinates_for_regular_n-dimensional_simplex_in_Rn */
  for (i = 0; i < dim; i++)
    {
      double tot2 = 0;
      int j;
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

  /* We scale up the caging simplex such that the radius of the inscribed sphere
     is larger than the maximum extent in any direction (which is .5 in this
     scaled space). The inscribed spheres radius is the altitude of the simplex
     divided by d+1 (see http://math.stackexchange.com/a/165433 or
     http://math.stackexchange.com/a/165390 for instance). */
  double radius;
  {
    double altitude = (gsl_matrix_get(tree->seed_points, 0, 0)
                       - gsl_matrix_get(tree->seed_points, 1, 0));
    radius = altitude/(dim+1);
  }

  /* We then scale it up by a factor of 1000.  This ensures that the method is
     robust to moderate outliers (points that are not within the min to max
     range as derived from the data or the min and max parameters).  This is
     arbitrary, but a useful safety net. */
  gsl_matrix_scale(tree->seed_points, 1000/radius);

  /* We also apply the inverse of the shift and scale to these points as it
     simplifies the implementation. */
  for (i = 0; i < tree->dim+1; i++)
    {
      gsl_vector_view v = gsl_matrix_row(tree->seed_points, i);
      gsl_vector_div(&(v.vector), tree->scale);
      gsl_vector_add(&(v.vector), tree->shift);
    }

  for (i = 0; i < tree->dim+1; i++)
    {
      tree->root->points[i] = -(i+1);
    }

  for (i = 0; i < tree->root->n_links; i++)
    {
      /* This is a special triangle that doesn't have neighbors to consider */
      tree->root->links[i] = NULL;
    }

  gsl_permutation_init(tree->shuffle);
  if (rng)
    gsl_ran_shuffle(rng, tree->shuffle->data, data->size1, sizeof(size_t));

  int ret = GSL_SUCCESS;
  if (data)
    {
      for (i = 0; i < data->size1; i++)
        {
          gsl_vector_view new_point
            = gsl_matrix_row(data, gsl_permutation_get(tree->shuffle, i));
          simplex_tree_node *leaf
            = find_leaf(tree, data, &(new_point.vector), tree->accel);
          ret = insert_point(tree, leaf, data, &(new_point.vector),
                             tree->accel);
          if (GSL_SUCCESS != ret)
            break;
        }
    }
  return ret;
}

simplex_tree_accel *
simplex_tree_accel_alloc(int dim)
{
  simplex_tree_accel *accel = malloc(sizeof(simplex_tree_accel));
  accel->simplex_matrix = gsl_matrix_alloc(dim, dim);
  accel->perm = gsl_permutation_alloc(dim);
  accel->coords = gsl_vector_alloc(dim);
  accel->current_simplex = NULL;
}

void
simplex_tree_accel_free(simplex_tree_accel *accel)
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
  if (!accel) local_accel = tree->accel;

  simplex_tree_node *ret;
  if (contains_point(tree, tree->root, data, point, local_accel))
    ret = _find_leaf(tree, tree->root, data, point, local_accel);
  else
    /* Outside the cage.  I should handle this more gracefully.  Insuring that a
       point interpolation will return zero and that a point insertion will be
       meaningful would be enough. */
    assert(0);

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
  if (!accel) accel = tree->accel;
  assert(leaf->leaf_p);
  leaf->leaf_p = 0;
  int dim = tree->dim;

  simplex_tree_node **new_simplexes = tree->new_simplexes;

  /* Allocate $d$ new simplexes */
  int ismplx;
  for (ismplx = 0; ismplx < dim + 1; ismplx++)
    {
      new_simplexes[ismplx] = simplex_tree_node_alloc(dim);
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

  tree->n_points++;

  struct node_list *seen = NULL;
  check_leaf_nodes(tree, leaf->links[0], &seen, NULL);
  free_list(seen); seen = NULL;

  /* Now fix the Delaunay condition */
  for (i = 0; i < leaf->n_links; i++)
    {
      /* Only check if this is a leaf (if it isn't a leaf then we have surely
         already looked at it) */
      if (!leaf->links[i]->leaf_p) continue;
      delaunay(tree, leaf->links[i], data, 0, accel);
      check_leaf_nodes(tree, leaf->links[0], &seen, NULL);
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

  int ret = 0;
  if (in_hypersphere(tree, leaf, data, neighbor->points[far], accel))
    {
      ret = 1;
      /* This should always be true as you will never flip an edge in 1D */
      assert(dim > 1);

      /* We need to "flip" this face. */
      leaf->leaf_p = 0;
      neighbor->leaf_p = 0;
      leaf->flipped = 1;
      neighbor->flipped = 1;

      simplex_tree_node **new_simplexes = tree->new_simplexes;
      simplex_tree_node **old_neighbors1 = tree->old_neighbors1;
      simplex_tree_node **old_neighbors2 = tree->old_neighbors2;
      int *left_out = tree->left_out;

      /* Save current neighbors */
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
          new_simplexes[ismplx] = simplex_tree_node_alloc(dim);
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
      leaf->links[dim] = neighbor;
      neighbor->links[dim] = leaf;

      /* Now, recursively check the new external faces as they could now also
         need edge flips */
      for (ismplx = 0; ismplx < dim; ismplx++)
        for (i = 0; i < dim+1; i++)
          {
            if (!leaf->links[ismplx]->links[i]) continue;
            delaunay(tree, leaf->links[ismplx]->links[i], data, i, accel);
            if (leaf->links[ismplx]->flipped) break;
          }
    }

  return ret;
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
    point = gsl_matrix_row(data, gsl_permutation_get(tree->shuffle, idx));

  calculate_hypersphere(tree, node, data, x0, &r2, accel);

  /* Compute the square magnitude of displacement */
  double dist2 = 0;
  int i;
  for (i = 0; i < tree->dim; i++)
    {
      double comp = (gsl_vector_get(tree->scale, i)
                     * (gsl_vector_get(&(point.vector), i)
                        - gsl_vector_get(tree->shift, i)));
      double val = comp - gsl_vector_get(x0, i);
      dist2 += val*val;
    }

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
        vi = gsl_matrix_row(data, gsl_permutation_get(tree->shuffle, vi_idx));

      if (vi1_idx < 0)
        vi1 = gsl_matrix_row(tree->seed_points, -vi1_idx - 1);
      else
        vi1 = gsl_matrix_row(data, gsl_permutation_get(tree->shuffle, vi1_idx));

      for (j = 0; j < dim; j++)
        {
          double pij = (gsl_vector_get(tree->scale, j)
                        * (gsl_vector_get(&(vi.vector), j)
                           - gsl_vector_get(tree->shift, j)));
          double pij1 = (gsl_vector_get(tree->scale, j)
                        * (gsl_vector_get(&(vi1.vector), j)
                           - gsl_vector_get(tree->shift, j)));
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
    first_point = gsl_matrix_row(data, gsl_permutation_get(tree->shuffle, idx));

  gsl_vector_memcpy(accel->coords, &(first_point.vector));
  /* Why doesn't this work? */
  /* gsl_blas_daxpy(-1, x0, accel->coords); */
  gsl_vector_sub(accel->coords, tree->shift);
  gsl_vector_mul(accel->coords, tree->scale);
  gsl_vector_sub(accel->coords, x0);
  *r2 = dnrm22(accel->coords);

  return GSL_SUCCESS;
}

int
calculate_bary_coords(simplex_tree *tree, simplex_tree_node *node, gsl_matrix *data,
                      gsl_vector *point,
                      simplex_tree_accel *accel)
{
  if (!accel) accel = tree->accel;
  int dim = tree->dim;
  gsl_vector_view x0;
  int x0_idx = node->points[dim];

  if (x0_idx < 0)
    x0 = gsl_matrix_row(tree->seed_points, -x0_idx - 1);
  else
    x0 = gsl_matrix_row(data, gsl_permutation_get(tree->shuffle, x0_idx));

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
            p = gsl_matrix_row(data, gsl_permutation_get(tree->shuffle, xi_idx));
          for (j = 0; j < dim; j++)
            {
              double pv = (gsl_vector_get(tree->scale, j)
                           * (gsl_vector_get(&(p.vector), j)
                              - gsl_vector_get(tree->shift, j)));
              double xv = (gsl_vector_get(tree->scale, j)
                           * (gsl_vector_get(&(x0.vector), j)
                              - gsl_vector_get(tree->shift, j)));
              gsl_matrix_set(accel->simplex_matrix, j, i, pv-xv);
            }
        }

      int signum;
      gsl_linalg_LU_decomp(accel->simplex_matrix, accel->perm, &signum);
    }

  gsl_vector *pp = gsl_vector_alloc(dim);
  gsl_vector_memcpy(pp, point);
  gsl_vector_sub(pp, &(x0.vector));
  gsl_vector_mul(pp, tree->scale);

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
        {
          xi_idx = gsl_permutation_get(tree->shuffle, xi_idx);
          interp += coord * gsl_vector_get(response, xi_idx);
        }
    }
  int idx = leaf->points[dim];
  /* Only add a contribution if the point isn't a seed point */
  if (idx >= 0)
    {
      idx = gsl_permutation_get(tree->shuffle, idx);
      interp += (1 - tot) * gsl_vector_get(response, idx);
    }
  return interp;
}
