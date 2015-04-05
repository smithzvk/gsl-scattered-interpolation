
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

typedef struct simplex_tree_struct
{
  simplex_tree_node *root;
  gsl_matrix *seed_points;
  int n_points;
  int dim;
} simplex_tree;

typedef struct
{
  gsl_matrix *simplex_matrix;
  gsl_permutation *perm;
  gsl_vector *coords;
  simplex_tree_node *current_simplex;
} simplex_tree_accel;

simplex_tree_node * alloc_simplex_tree_node(int dim);

simplex_tree * alloc_simplex_tree(int dim);

void free_simplex_tree(simplex_tree *tree);

void free_simplex_tree_node(simplex_tree *tree, simplex_tree_node *node);

simplex_tree_accel *alloc_simplex_tree_accel(size_t dim);

void free_simplex_tree_accel(simplex_tree_accel *accel);

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

int build_triangulation(simplex_tree *tree,
                        gsl_matrix *data,
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
