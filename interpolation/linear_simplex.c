
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

simplex_index
simplex_tree_node_alloc(simplex_tree *tree)
{
  int i;
  int dim = tree->dim;

  if (tree->n_simplexes + 1 >= tree->max_simplexes)
    {
      tree->max_simplexes *= 2;
      tree->simplexes = realloc(tree->simplexes, (tree->max_simplexes
                                                  * sizeof(simplex_tree_node)));
    }
  simplex_tree_node *node = &(tree->simplexes[tree->n_simplexes]);

  if (tree->n_pidx + dim + 1 >= tree->max_pidx)
    {
      tree->max_pidx *= 2;
      tree->pidx = realloc(tree->pidx, tree->max_pidx * sizeof(int));
    }
  node->points = tree->n_pidx;
  tree->n_pidx += dim + 1;

  if (tree->n_links + dim + 1 >= tree->max_links)
    {
      tree->max_links *= 2;
      tree->links = realloc(tree->links, (tree->max_links
                                          * sizeof(simplex_index)));
    }
  node->links = tree->n_links;
  tree->n_links += dim + 1;

  node->leaf_p = 1;
  node->flipped = 0;

  return tree->n_simplexes++;
}

simplex_tree *
simplex_tree_alloc(int dim, int n_points)
{
  simplex_tree *tree = malloc(sizeof(simplex_tree));
  tree->dim = dim;
  tree->seed_points = gsl_matrix_alloc(dim+1, dim);
  tree->n_points = 0;
  tree->max_points = n_points;

  /* A multiplicative factor to account for internal nodes */
  int overhead = 9;

  /* Allocate the arrays */
  tree->max_pidx = overhead * n_points * (dim+1);
  tree->pidx = malloc(tree->max_pidx * sizeof(int));
  tree->n_pidx = 0;

  tree->max_links = overhead * n_points * (dim+1);
  tree->links = malloc(tree->max_links * sizeof(simplex_index));
  tree->n_links = 0;

  tree->max_simplexes = overhead * n_points;
  tree->simplexes = malloc(tree->max_simplexes * sizeof(simplex_tree_node));
  tree->n_simplexes = 0;

  /* Allocate the root node */
  simplex_index node = simplex_tree_node_alloc(tree);

  /* Builtin accelerator */
  tree->accel = simplex_tree_accel_alloc(dim);

  /* Memory for inserting points and flipping edges */
  tree->new_simplexes = malloc((dim+1) * sizeof(simplex_index));
  tree->old_neighbors1 = malloc(dim * sizeof(simplex_index));
  tree->old_neighbors2 = malloc(dim * sizeof(simplex_index));
  tree->left_out = malloc(dim * sizeof(int));

  /* Memory for transforming the data */
  tree->shift = gsl_vector_alloc(dim);
  tree->scale = gsl_vector_alloc(dim);
  tree->min = gsl_vector_alloc(dim);
  tree->max = gsl_vector_alloc(dim);

  /* Random data permutation */
  tree->shuffle = gsl_permutation_alloc(n_points);

  /* Working memory for various purposes */
  tree->tmp_vec = gsl_vector_alloc(dim);

  return tree;
}

void
simplex_tree_free(simplex_tree *tree)
{
  gsl_matrix_free(tree->seed_points);
  simplex_tree_accel_free(tree->accel);
  free(tree->simplexes);
  free(tree->pidx);
  free(tree->links);

  free(tree->new_simplexes);
  free(tree->old_neighbors1);
  free(tree->old_neighbors2);
  free(tree->left_out);
  gsl_vector_free(tree->shift);
  gsl_vector_free(tree->scale);
  gsl_vector_free(tree->min);
  gsl_vector_free(tree->max);
  gsl_permutation_free(tree->shuffle);
  gsl_vector_free(tree->tmp_vec);

  free(tree);
}

int
simplex_tree_init(simplex_tree *tree, gsl_matrix *data,
                  gsl_vector *min, gsl_vector *max,
                  int init_flags, gsl_rng *rng)
{
  int i, dim = tree->dim;

  if (!(data || (min && max) || (init_flags & SIMPLEX_TREE_NOSTANDARDIZE)))
    {
      /* You need to either provide a min and max, or provide some
         representative data to infer one from, or tell the algorithm to not
         standardize with SIMPLEX_TREE_NOSTANDARDIZE. */
      return GSL_FAILURE;
    }
  else if (init_flags & SIMPLEX_TREE_NOSTANDARDIZE)
    {
      for (i = 0; i < dim; i++)
        {
          gsl_vector_set(tree->min, i, -0.5);
          gsl_vector_set(tree->max, i, +0.5);
        }
    }
  else if (data && (!min || !max))
    {
      /* Fill in what we are given */
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

      /* Infer the rest from the given data */
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
    }

  /* Calculate the shift and scale */
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

  /* Make scaling isotropic if asked (shift remains independent) */
  if (!(SIMPLEX_TREE_NOSTANDARDIZE & init_flags)
      && (SIMPLEX_TREE_ISOSCALE & init_flags))
    {
      double min_comp = gsl_vector_get(tree->scale, 0);
      for (i = 1; i < dim; i++)
        {
          if (min_comp > gsl_vector_get(tree->scale, i))
            min_comp = gsl_vector_get(tree->scale, i);
        }
      for (i = 0; i < dim; i++)
          gsl_vector_set(tree->scale, i, min_comp);
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

  /* We then scale it up by a factor dependent on the machine precision
     (1/GSL_ROOT5_DBL_EPSILON).  This ensures that the method is robust to
     moderate outliers (points that are not within the min to max range as
     derived from the data or the min and max parameters).  This is an
     arbitrary but a useful safety net. */
  gsl_matrix_scale(tree->seed_points, 1/(GSL_ROOT5_DBL_EPSILON * radius));

  /* We also apply the inverse of the shift and scale to these points as it
     simplifies the implementation. */
  for (i = 0; i < dim+1; i++)
    {
      gsl_vector_view v = gsl_matrix_row(tree->seed_points, i);
      gsl_vector_div(&(v.vector), tree->scale);
      gsl_vector_add(&(v.vector), tree->shift);
    }

  for (i = 0; i < dim+1; i++)
    POINT(0, i) = -(i+1);

  for (i = 0; i < dim+1; i++)
    /* This is a special triangle that doesn't have neighbors to consider */
    LINK(0, i) = 0;

  gsl_permutation_init(tree->shuffle);

  int ret = GSL_SUCCESS;
  if (data)
    {
      if (tree->n_points + data->size1 > tree->max_points)
        {
          /* Not enough room in tree for these points */
          return GSL_FAILURE;
        }

      if (rng)
        gsl_ran_shuffle(rng, tree->shuffle->data, data->size1, sizeof(size_t));

      for (i = 0; i < data->size1; i++)
        {
          gsl_vector_view new_point
            = gsl_matrix_row(data, gsl_permutation_get(tree->shuffle, i));
          simplex_index leaf
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
  accel->current_simplex = -1;
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
point_in_simplex(simplex_tree *tree, simplex_index node, int point)
{
  int dim = tree->dim;
  int i;
  for (i = 0; i < dim + 1; i++)
    {
      if (POINT(node, i) == point)
        break;
    }
  return (i < dim + 1);
}

simplex_index
find_leaf(simplex_tree *tree, gsl_matrix *data,
          gsl_vector *point,
          simplex_tree_accel *accel)
{
  simplex_tree_accel *local_accel = accel;
  int dim = tree->dim;
  if (!accel) local_accel = tree->accel;

  simplex_index ret;
  if (contains_point(tree, 0, data, point, local_accel))
    ret = _find_leaf(tree, 0, data, point, local_accel);
  else
    /* Outside the cage.  I should handle this more gracefully.  Insuring that a
       point interpolation will return zero and that a point insertion will be
       meaningful would be enough. */
    assert(("Given point outside the domain", 0));

  return ret;
}

simplex_index
_find_leaf(simplex_tree *tree, simplex_index node, gsl_matrix *data,
           gsl_vector *point,
           simplex_tree_accel *accel)
{
  if (SIMP(node)->leaf_p)
    {
      return node;
    }
  else
    {
      int i;
      int dim = tree->dim;
      for (i = 0; i < 1 + dim - SIMP(node)->flipped; i++)
        {
          if (LINK(node, i)
              && contains_point(tree, LINK(node, i), data, point, accel))
            {
              return _find_leaf(tree, LINK(node, i), data, point, accel);
            }
        }
    }
  /* We should never get here.  What if something falls through the cracks.
     This can happen if the point lies on a face.  Perhaps I could find the
     simplex that is the closest match and use that if none of them match.  This
     would be a useful fall-back. */
  assert(("No subsimplex contains point, floating point precision issue?", 0));
}

int
insert_point(simplex_tree *tree, simplex_index leaf,
             gsl_matrix *data, gsl_vector *point,
             simplex_tree_accel *accel)
{
  int i, j;
  if (!accel) accel = tree->accel;
  assert(("You can only insert a point into a leaf", SIMP(leaf)->leaf_p));
  SIMP(leaf)->leaf_p = 0;
  int dim = tree->dim;

  simplex_index *new_simplexes = tree->new_simplexes;

  /* Allocate $d$ new simplexes */
  int ismplx;
  for (ismplx = 0; ismplx < dim + 1; ismplx++)
    {
      new_simplexes[ismplx] = simplex_tree_node_alloc(tree);
    }

  /* Populate the indexes in the newly created leaf nodes */
  for (i = 0; i < dim+1; i++)
    {
      POINT(new_simplexes[i], 0) = tree->n_points;
      int k = 1;
      for (j = 0; j < dim+1; j++)
        {
          if (j==i) continue;
          POINT(new_simplexes[i], k++) = POINT(leaf, j);
        }
    }

  /* Populate external links */
  for (i = 0; i < dim+1; i++)
    {
      simplex_index neighbor = LINK(leaf, i);
      assert(("Inconsistency found in simplex tree structure",
              !neighbor || SIMP(neighbor)->leaf_p));
      if (neighbor)
        assert(("Inconsistency found in simplex tree structure",
                !point_in_simplex(tree, neighbor, POINT(leaf, i))));

      /* By convention, the 0-face is the one that already existed */
      LINK(new_simplexes[i], 0) = neighbor;

      /* Fix the neighbor links for the existing leaf nodes */
      if (neighbor)
        {
          for (j = 0; j < dim+1; j++)
            if (LINK(neighbor, j) == leaf)
              break;
          assert(("Inconsistency found in simplex tree structure, "
                  "no reverse link",
                  j < dim+1));
          LINK(neighbor, j) = new_simplexes[i];
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
                                    POINT(new_simplexes[ismplx],i)))
                break;
            }
          assert(("Inconsistency found in simplex tree structure, "
                  "proper internal simplex not found",
                  jsmplx < dim+1));
          LINK(new_simplexes[ismplx], i) = new_simplexes[jsmplx];
        }
    }

  for (i = 0; i < dim+1; i++)
    LINK(leaf, i) = new_simplexes[i];

  tree->n_points++;

  /* Now fix the Delaunay condition */
  for (i = 0; i < dim+1; i++)
    {
      /* Only check if this is a leaf (if it isn't a leaf then we have surely
         already looked at it) */
      if (!(SLINK(leaf, i)->leaf_p)) continue;
      delaunay(tree, LINK(leaf, i), data, 0, accel);
    }
  check_delaunay(tree, data);

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
delaunay(simplex_tree *tree, simplex_index leaf,
         gsl_matrix *data,
         /* face is the index in the simplex that identifies the face */
         int face,
         simplex_tree_accel *accel)
{
  int i;

  int dim = tree->dim;

  if (!LINK(leaf, face)) return GSL_SUCCESS;
  assert(("Checking if a flip is necessary on an already flipped simplex",
          !SIMP(leaf)->flipped));
  assert(("Checking if a flip is necessary on a non-leaf simplex",
          SIMP(leaf)->leaf_p));

  simplex_index neighbor = LINK(leaf, face);
  assert(("Inconsistency found in simplex tree structure, "
          "neighbor to leaf not a leaf",
          SIMP(neighbor)->leaf_p));

  /* Find far point */
  int far;
  for (far = 0; far < dim+1; far++)
    {
      if (LINK(neighbor, far) == leaf)
        break;
    }
  assert(("Inconsistency found in simplex tree structure, "
          "reverse link not found",
          far < dim+1));

  int ret = 0;
  if (in_hypersphere(tree, leaf, data, POINT(neighbor, far), accel))
    {
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
      int *left_out = tree->left_out;

      /* Save current neighbors */
      {
        int k = 0;
        int ok = 0;
        for (i = 0; i < dim+1; i++)
          if (LINK(leaf, i) != neighbor)
            old_neighbors1[k++] = LINK(leaf, i);
          else
            ok++;
        assert(("Inconsistency found in simplex tree structure, "
                "didn't find one and only one reverse link",
                ok == 1));
        ok = 0;
        k = 0;
        for (i = 0; i < dim+1; i++)
          if (LINK(neighbor, i) != leaf)
            old_neighbors2[k++] = LINK(neighbor, i);
          else
            ok++;
        assert(("Inconsistency found in simplex tree structure, "
                "didn't find one and only one reverse link",
                ok == 1));
      }

      /* Allocate new simplexes */
      int ismplx;
      for (ismplx = 0; ismplx < dim; ismplx++)
        {
          new_simplexes[ismplx] = simplex_tree_node_alloc(tree);
        }

      /* Set points on simplexes */
      for (ismplx = 0; ismplx < dim; ismplx++)
        {
          POINT(new_simplexes[ismplx], 0) = POINT(leaf, face);
          POINT(new_simplexes[ismplx], 1) = POINT(neighbor, far);

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

              POINT(new_simplexes[ismplx], idx_on_face+2) = POINT(leaf, j);
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
                                    POINT(leaf, left_out[ismplx])))
                break;
            }
          assert(("Inconsistency found in simplex tree structure, "
                  "no proper external link found",
                  jsmplx < dim || null_option));

          {
            simplex_index ext;
            if (jsmplx < dim)
              ext = LINK(new_simplexes[ismplx], 0) = old_neighbors2[jsmplx];
            else
              ext = LINK(new_simplexes[ismplx], 0) = 0;

            if (ext)
              {
                int j;
                for (j = 0; j < dim+1; j++)
                  if (LINK(ext, j) == neighbor)
                    break;
                assert(("Inconsistency found in simplex tree structure, "
                        "no reverse link found",
                        j < dim+1));
                LINK(ext, j) = new_simplexes[ismplx];
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
                                    POINT(leaf, left_out[ismplx])))
                break;
            }
          assert(("Inconsistency found in simplex tree structure, "
                  "no proper external link found",
                  jsmplx < dim || null_option));

          {
            simplex_index ext;
            if (jsmplx < dim)
              ext = LINK(new_simplexes[ismplx], 1) = old_neighbors1[jsmplx];
            else
              ext = LINK(new_simplexes[ismplx], 1) = 0;

            if (ext)
              {
                int j;
                for (j = 0; j < dim+1; j++)
                  if (LINK(ext, j) == leaf)
                    break;
                assert(("Inconsistency found in simplex tree structure, "
                        "no reverse link found",
                        j < dim+1));
                LINK(ext, j) = new_simplexes[ismplx];
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
                                        POINT(new_simplexes[ismplx], i)))
                    break;
                }
              assert(("Inconsistency found in simplex tree structure, "
                      "proper internal simplex not found",
                      jsmplx < dim+1));
              LINK(new_simplexes[ismplx], i) = new_simplexes[jsmplx];
            }
        }

      for (i = 0; i < dim; i++)
        {
          LINK(leaf, i) = new_simplexes[i];
          LINK(neighbor, i) = new_simplexes[i];
        }
      LINK(leaf, dim) = neighbor;
      LINK(neighbor, dim) = leaf;

      /* Now, recursively check the new external faces as they could now also
         need edge flips */
      for (ismplx = 0; ismplx < dim; ismplx++)
        for (i = 0; i < dim+1; i++)
          {
            if (!LINK(LINK(leaf, ismplx), i)) continue;
            delaunay(tree, LINK(leaf, ismplx), data, i, accel);
            if (SLINK(leaf, ismplx)->flipped) break;
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
in_hypersphere(simplex_tree *tree, simplex_index node,
               gsl_matrix *data,
               int idx, simplex_tree_accel *accel)
{
  gsl_vector *x0 = tree->tmp_vec;
  double r2;

  gsl_vector_view point;
  if (idx < 0)
    point = gsl_matrix_row(tree->seed_points, -idx - 1);
  else
    point = gsl_matrix_row(data, gsl_permutation_get(tree->shuffle, idx));

  /* If we cannot compute the hypersphere for any reason assume it is because
     the points are degenerate (do not span the dimensionality). */
  if (GSL_SUCCESS != calculate_hypersphere(tree, node, data, x0, &r2, accel))
      return 1;

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

  /* Small correction to the radius to remove degenerate cases */
  return dist2 < (r2 * (1 - GSL_SQRT_DBL_EPSILON));
}

static int
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

/* See http://steve.hollasch.net/cgindex/geometry/sphere4pts.html (in
   particular, Dr. John S. Eickemeyer's suggestion) on how to do this in a way
   that generalizes to arbitrary dimensionality. */
int
calculate_hypersphere(simplex_tree *tree, simplex_index node,
                      gsl_matrix *data,
                      gsl_vector *x0, double *r2,
                      simplex_tree_accel *accel)
{
  int i, j;
  int dim = tree->dim;
  accel->current_simplex = -1;

  for (i = 0; i < dim; i++)
    {
      gsl_vector_set(accel->coords, i, 0);

      gsl_vector_view vi, vi1;
      int vi_idx = POINT(node, i);
      int vi1_idx = POINT(node, i+1);

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
  if (singular(accel->simplex_matrix))
    return GSL_FAILURE;

  gsl_linalg_LU_solve(accel->simplex_matrix, accel->perm, accel->coords, x0);


  gsl_vector_view first_point;
  int idx = POINT(node, 0);
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
calculate_bary_coords(simplex_tree *tree, simplex_index node, gsl_matrix *data,
                      gsl_vector *point,
                      simplex_tree_accel *accel)
{
  if (!accel) accel = tree->accel;
  int dim = tree->dim;
  gsl_vector_view x0;
  int x0_idx = POINT(node, dim);

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
          int xi_idx = POINT(node, i);
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

  gsl_vector *pp = tree->tmp_vec;
  gsl_vector_memcpy(pp, point);
  gsl_vector_sub(pp, &(x0.vector));
  gsl_vector_mul(pp, tree->scale);

  gsl_linalg_LU_solve(accel->simplex_matrix, accel->perm, pp, accel->coords);
  return GSL_SUCCESS;
}

int
contains_point(simplex_tree *tree, simplex_index node,
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
interp_point(simplex_tree *tree, simplex_index leaf,
             gsl_matrix *data, gsl_vector *response, gsl_vector *point,
             simplex_tree_accel *accel)
{
  int i;
  int dim = tree->dim;

  assert(("Interpolation must be on a leaf node", SIMP(leaf)->leaf_p));
  calculate_bary_coords(tree, leaf, data, point, accel);

  double tot = 0;
  double interp = 0;
  for (i = 0; i < dim; i++)
    {
      double coord = gsl_vector_get(accel->coords, i);
      tot += coord;
      int xi_idx = POINT(leaf, i);
      /* Only add a contribution if the point isn't a seed point */
      if (xi_idx >= 0)
        {
          xi_idx = gsl_permutation_get(tree->shuffle, xi_idx);
          interp += coord * gsl_vector_get(response, xi_idx);
        }
    }
  int idx = POINT(leaf, dim);
  /* Only add a contribution if the point isn't a seed point */
  if (idx >= 0)
    {
      idx = gsl_permutation_get(tree->shuffle, idx);
      interp += (1 - tot) * gsl_vector_get(response, idx);
    }
  return interp;
}
