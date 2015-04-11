#include <gsl/gsl_rng.h>

typedef struct simplex_tree_node_struct
{
  int *points;
  /* An array to pointers to other trees contained in this one.  NULL if this is
     a leaf. */
  struct simplex_tree_node_struct **links;
  /* Mark if this is a leaf or not */
  char leaf_p, flipped;
  /* Extents O(d) numbers*/
  char n_links;
} simplex_tree_node;

typedef struct
{
  gsl_matrix *simplex_matrix;
  gsl_permutation *perm;
  gsl_vector *coords;
  simplex_tree_node *current_simplex;
} simplex_tree_accel;

typedef struct simplex_tree_struct
{
  simplex_tree_node *root;
  gsl_matrix *seed_points;
  int n_points;
  int dim;
  gsl_vector *shift;
  gsl_vector *scale;
  gsl_vector *min;
  gsl_vector *max;
  gsl_permutation *shuffle;
  simplex_tree_accel *accel;
  simplex_tree_node **new_simplexes;
  simplex_tree_node **old_neighbors1;
  simplex_tree_node **old_neighbors2;
  int *left_out;
} simplex_tree;

simplex_tree_node * simplex_tree_node_alloc(int dim);

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

void simplex_tree_node_free(simplex_tree *tree, simplex_tree_node *node);

simplex_tree_accel * simplex_tree_accel_alloc(int dim);

void simplex_tree_accel_free(simplex_tree_accel *accel);

int point_in_simplex(simplex_tree *tree, simplex_tree_node *node, int point);

simplex_tree_node *find_leaf(simplex_tree *tree, gsl_matrix * data,
                             gsl_vector *point,
                             simplex_tree_accel *accel);

simplex_tree_node *_find_leaf(simplex_tree *tree, simplex_tree_node *node,
                              gsl_matrix * data,
                              gsl_vector *point,
                              simplex_tree_accel *accel);

int insert_point(simplex_tree *tree, simplex_tree_node *leaf,
                 gsl_matrix *data, gsl_vector *point,
                 simplex_tree_accel *accel);

int delauney(simplex_tree *tree, simplex_tree_node *leaf,
             gsl_matrix *data, simplex_tree_accel *accel);

int in_hypersphere(simplex_tree *tree, simplex_tree_node *node,
                   gsl_matrix *data,
                   int idx, simplex_tree_accel *accel);

int calculate_hypersphere(simplex_tree *tree, simplex_tree_node *node,
                          gsl_matrix *data,
                          gsl_vector *x0, double *r2,
                          simplex_tree_accel *accel);

int calculate_bary_coords(simplex_tree *tree, simplex_tree_node *node,
                          gsl_matrix *data,
                          gsl_vector *point,
                          simplex_tree_accel *accel);


int contains_point(simplex_tree *tree, simplex_tree_node *node,
                   gsl_matrix *data,
                   gsl_vector *point,
                   simplex_tree_accel *accel);

double interp_point(simplex_tree *tree, simplex_tree_node *_leaf,
                    gsl_matrix *data,
                    gsl_vector *response,
                    gsl_vector *point,
                    simplex_tree_accel *accel);
