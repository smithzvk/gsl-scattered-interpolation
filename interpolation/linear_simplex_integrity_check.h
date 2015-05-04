
#ifndef __GSL_LINEAR_SIMPLEX_INTEGRITY_CHECK_H__
#define __GSL_LINEAR_SIMPLEX_INTEGRITY_CHECK_H__

struct node_list
{
  simplex_index val;
  struct node_list *next;
};

void check_leaf_nodes(simplex_tree *tree,
                      void (*fn)(simplex_tree *, simplex_index));
int in_list(simplex_index node, struct node_list *list);
int cycle(struct node_list *list);
void free_list(struct node_list *list);

int check_delaunay(simplex_tree *tree, gsl_matrix *data);
void output_triangulation(simplex_tree *tree, gsl_matrix *data, gsl_vector *response,
                          char lines_filename[], char points_filename[],
                          char circles_filename[]);

#endif
