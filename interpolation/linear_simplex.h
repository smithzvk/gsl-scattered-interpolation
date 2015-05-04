#include <gsl/gsl_rng.h>

#ifndef __GSL_LINEAR_SIMPLEX_H__
#define __GSL_LINEAR_SIMPLEX_H__

typedef int simplex_index;

typedef struct simplex_tree_node_struct
{
  int points;
  simplex_index links;
  /* Flags marking if this is a leaf or flipped */
  unsigned int leaf_p:1, flipped:1;
} simplex_tree_node;

typedef struct
{
  gsl_matrix *simplex_matrix;
  gsl_permutation *perm;
  gsl_vector *coords;
  simplex_index current_simplex;
} simplex_tree_accel;

typedef struct simplex_tree_struct
{
  int n_simplexes, max_simplexes;
  simplex_tree_node *simplexes;

  int n_pidx, max_pidx;
  int *pidx;

  int n_links, max_links;
  simplex_index *links;

  gsl_matrix *seed_points;
  int n_points;
  int max_points;
  int dim;
  gsl_vector *shift;
  gsl_vector *scale;
  gsl_vector *min;
  gsl_vector *max;
  gsl_permutation *shuffle;
  simplex_tree_accel *accel;
  simplex_index *new_simplexes;
  simplex_index *old_neighbors1;
  simplex_index *old_neighbors2;
  int *left_out;
  int *tmp_points;
  gsl_vector *tmp_vec1, *tmp_vec2;
  gsl_matrix *tmp_mat;
} simplex_tree;

#define SIMP(I) (&(tree->simplexes[(I)]))
#define LINK(NODE, I) (tree->links[(I) + SIMP(NODE)->links])
#define SLINK(NODE, I) (SIMP(LINK(NODE, I)))
#define POINT(NODE, I) (tree->pidx[(I) + SIMP(NODE)->points])

#define DATA_POINT(DATA, POINT) _data_point(tree, (DATA), (POINT))

static inline gsl_vector_view
_data_point(simplex_tree *tree, gsl_matrix *data, int point)
{
  gsl_vector_view view;
  if (point < 0)
    view = gsl_matrix_row(tree->seed_points, -(point) - 1);
  else
    view = gsl_matrix_row(data, gsl_permutation_get(tree->shuffle, point));

  return view;
}

#define FIND(VAR, PRED, ...)                            \
  for (VAR = 0; VAR < tree->dim+1; VAR++)               \
    {                                                   \
      if (PRED)                                         \
        break;                                          \
    }                                                   \
  assert(("Couldn't satisfy predicate: ",               \
          PRED, ""__VA_ARGS__"", VAR < tree->dim+1));


simplex_index simplex_tree_node_alloc(simplex_tree *tree);

simplex_tree * simplex_tree_alloc(int dim, int n_points);

#define SIMPLEX_TREE_DEFAULT 0
/* #define SIMPLEX_TREE_NORANDOMIZE (1 << 0) */
#define SIMPLEX_TREE_NOSTANDARDIZE (1 << 0)
#define SIMPLEX_TREE_ISOSCALE (1 << 1)
/* This should only be used when doing interpolation, should it should be part
   of the initialization of that workspace. */
/* #define SIMPLEX_TREE_REMOVE_LINEAR */

int simplex_tree_init(simplex_tree *tree, gsl_matrix *data,
                      gsl_vector *min, gsl_vector *max,
                      int init_flags, gsl_rng *rng);


void simplex_tree_free(simplex_tree *tree);

simplex_tree_accel * simplex_tree_accel_alloc(int dim);

void simplex_tree_accel_free(simplex_tree_accel *accel);

int point_in_simplex(simplex_tree *tree, simplex_index node, int point);

simplex_index find_leaf(simplex_tree *tree, gsl_matrix * data,
                        gsl_vector *point,
                        simplex_tree_accel *accel);

simplex_index _find_leaf(simplex_tree *tree, simplex_index node,
                         gsl_matrix * data,
                         gsl_vector *point,
                         simplex_tree_accel *accel);

int insert_point(simplex_tree *tree, simplex_index leaf,
                 gsl_matrix *data, gsl_vector *point,
                 simplex_tree_accel *accel);

int delauney(simplex_tree *tree, simplex_index leaf,
             gsl_matrix *data, simplex_tree_accel *accel);

int in_hypersphere(simplex_tree *tree, simplex_index node,
                   gsl_matrix *data,
                   int idx, simplex_tree_accel *accel);

int
in_hypersphere_points(simplex_tree *tree, int *points,
                      gsl_matrix *data,
                      int idx, simplex_tree_accel *accel);

int calculate_hypersphere(simplex_tree *tree, int *points,
                          gsl_matrix *data,
                          gsl_vector *x0, double *r2,
                          simplex_tree_accel *accel);

int calculate_bary_coords(simplex_tree *tree, simplex_index node,
                          gsl_matrix *data,
                          gsl_vector *point,
                          simplex_tree_accel *accel);


int contains_point(simplex_tree *tree, simplex_index node,
                   gsl_matrix *data,
                   gsl_vector *point,
                   simplex_tree_accel *accel);

double interp_point(simplex_tree *tree, simplex_index leaf,
                    gsl_matrix *data,
                    gsl_vector *response,
                    gsl_vector *point,
                    simplex_tree_accel *accel);

#endif
