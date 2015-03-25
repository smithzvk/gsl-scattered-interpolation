
typedef struct _simplex_tree_struct
{
  int *points;
  /* An array to pointers to other trees contained in this one.  NULL if this is
     a leaf. */
  struct _simplex_tree_struct **links;
  /* Mark if this is a leaf on not */
  char leaf_p;
  /* Extents O(d) numbers*/
  char n_points, n_links;
} _simplex_tree;

typedef struct simplex_tree_struct
{
  _simplex_tree *tree;
  gsl_matrix *seed_points;
  size_t n_points;
} simplex_tree;

typedef struct
{
  gsl_matrix *simplex_matrix;
  gsl_permutation *perm;
  gsl_vector *coords;
  _simplex_tree *current_simplex;
} simplex_tree_accel;

_simplex_tree * alloc_simplex_tree(int dim);

simplex_tree * initial_simplex_tree(int dim);

void free_simplex_tree(simplex_tree *tree);

void _free_simplex_tree(_simplex_tree *tree);

simplex_tree_accel *alloc_simplex_tree_accel(size_t dim);

void free_simplex_tree_accel(simplex_tree_accel *accel);

int point_in_simplex(simplex_tree *tree, _simplex_tree *node, int point);

_simplex_tree *find_leaf(simplex_tree *tree, gsl_matrix * data,
                         gsl_vector *point,
                         simplex_tree_accel *accel);

_simplex_tree *_find_leaf(simplex_tree *tree, _simplex_tree *_tree, gsl_matrix * data,
                          gsl_vector *point,
                          simplex_tree_accel *accel);

int insert_point(simplex_tree *tree, _simplex_tree *leaf,
                 gsl_matrix *data, gsl_vector *point,
                 simplex_tree_accel *accel);

int delauney(simplex_tree *tree, _simplex_tree *leaf,
             gsl_matrix *data, simplex_tree_accel *accel);

int in_hypersphere(simplex_tree *tree, _simplex_tree *_tree,
                   gsl_matrix *data,
                   int idx, simplex_tree_accel *accel);

int calculate_hypersphere(simplex_tree *tree, _simplex_tree *_tree,
                          gsl_matrix *data,
                          gsl_vector *x0, double *r2,
                          simplex_tree_accel *accel);

int calculate_bary_coords(simplex_tree *tree, _simplex_tree *_tree, gsl_matrix *data,
                          gsl_vector *point,
                          simplex_tree_accel *accel);


int contains_point(simplex_tree *tree, _simplex_tree *_tree,
                   gsl_matrix *data,
                   gsl_vector *point,
                   simplex_tree_accel *accel);

double interp_point(simplex_tree *tree, _simplex_tree *_leaf,
                    gsl_matrix *data,
                    gsl_vector *response,
                    gsl_vector *point,
                    simplex_tree_accel *accel);
