#include <assert.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_permutation.h>
#include <math.h>

#include "linear_simplex.h"

#include "linear_simplex_integrity_check.h"

struct node_list *
cons(void *car, struct node_list *cdr)
{
  struct node_list *new = malloc(sizeof(struct node_list));
  new->next = cdr;
  new->val = car;
  return new;
}

void
free_list(struct node_list *list)
{
  if (list->next)
    free_list(list->next);
  free(list);
}

int
cycle(struct node_list *list)
{
  struct node_list *lf, *ls;
  for (lf = list, ls = list;
       lf;
       lf = lf->next, ls = ls->next)
    {
      lf = lf->next;
      if (!lf) return 0;
      if (lf == ls) return 1;
    }
  return 0;
}

int
in_list(simplex_tree_node *node, struct node_list *list)
{
  if (!list)
    return 0;
  if (node == list->val)
    return 1;
  return in_list(node, list->next);
}

void
check_leaf_nodes(simplex_tree *tree, simplex_tree_node *node, struct node_list **seen,
                 void (*fn)(simplex_tree *, simplex_tree_node *))
{
  if (!node->leaf_p)
    {
      check_leaf_nodes(tree, node->links[0], seen, fn);
      return;
    }

  *seen = cons(node, *seen);

  if (fn) (*fn)(tree, node);
  int i;
  /* General leaf health */
  int j, k;
  for (j = 0; j < tree->dim+1; j++)
    for (k = j+1; k < tree->dim+1; k++)
      {
        assert(node->points[k] != node->points[j]);
        assert(node->links[k] != node);
        if (node->links[k]) assert(node->links[k] != node->links[j]);
      }

  for (i = 0; i < node->n_links; i++)
    {
      simplex_tree_node *neighbor = node->links[i];
      if (neighbor)
        {
          /* Check for forward and reverse linkage */
          for (k = 0; k < neighbor->n_links; k++)
            assert(node->points[i] != neighbor->points[k]);
          for (j = 0; j < neighbor->n_links; j++)
              if (neighbor->links[j] == node)
                break;
          assert(j < neighbor->n_links);
          for (k = 0; k < neighbor->n_links; k++)
            assert(node->points[k] != neighbor->points[j]);

          /* Basic sanity check, is the list corrupt */
          assert(!cycle(*seen));
          /* Recurse but only if we haven't seen this node yet */
          if (!in_list(neighbor, *seen))
            {
              check_leaf_nodes(tree, neighbor, seen, fn);
            }
        }
    }
}

gsl_matrix *gdata;

void
_check_delaunay(simplex_tree *tree, simplex_tree_node *node)
{
  int i;
  gsl_vector *x0 = gsl_vector_alloc(tree->dim);
  double r2;
  calculate_hypersphere(tree, node, gdata, x0, &r2, tree->accel);

  /* Check every point to see if any lie within the sphere. */
  gsl_vector *pp = gsl_vector_alloc(tree->dim);
  for (i = 0; i < tree->n_points; i++)
    {
      gsl_vector_view p = gsl_matrix_row(gdata, gsl_permutation_get(tree->shuffle, i));
      gsl_vector_memcpy(pp, &(p.vector));
      gsl_vector_sub(pp, tree->shift);
      gsl_vector_mul(pp, tree->scale);
      gsl_vector_sub(pp, x0);
      double mag2 = dnrm22(pp);
      assert(mag2 > r2 - 1e-3);
    }
  gsl_vector_free(pp);
  gsl_vector_free(x0);
}

int
check_delaunay(simplex_tree *tree, gsl_matrix *data)
{
  gdata = data;
  struct node_list *seen = NULL;
  check_leaf_nodes(tree, tree->root, &seen, _check_delaunay);
  free_list(seen); seen = NULL;
  return 1;
}

FILE *flines;
FILE *fcircles;
gsl_matrix *gdata;
gsl_vector *gresponse;

void
_output_triangulation(simplex_tree *tree, simplex_tree_node *node)
{
  int i;
  for (i = 0; i < tree->dim + 1; i++)
    {
      int i1 = node->points[i];
      int i2 = node->points[(i+1)%(tree->dim + 1)];

      gsl_vector_view p1, p2;
      double r1, r2;
      if (i1 < 0)
        {
          p1 = gsl_matrix_row(tree->seed_points, -i1 - 1);
          r1 = 0;
        }
      else
        {
          p1 = gsl_matrix_row(gdata, gsl_permutation_get(tree->shuffle, i1));
          if (gresponse)
            r1 = gsl_vector_get(gresponse, gsl_permutation_get(tree->shuffle, i1));
          else
            r1 = 0;
        }

      if (i2 < 0)
        {
          p2 = gsl_matrix_row(tree->seed_points, -i2 - 1);
          r2 = 0;
        }
      else
        {
          p2 = gsl_matrix_row(gdata, gsl_permutation_get(tree->shuffle, i2));
          if (gresponse)
            r2 = gsl_vector_get(gresponse, gsl_permutation_get(tree->shuffle, i2));
          else
            r2 = 0;
        }

      fprintf(flines,
              "%g %g %g\n%g %g %g\n\n\n",
              gsl_vector_get(tree->scale, 0)
              * (gsl_vector_get(&(p1.vector), 0)
                 - gsl_vector_get(tree->shift, 0)),
              gsl_vector_get(tree->scale, 1)
              * (gsl_vector_get(&(p1.vector), 1)
                 - gsl_vector_get(tree->shift, 1)),
              r1,
              gsl_vector_get(tree->scale, 0)
              * (gsl_vector_get(&(p2.vector), 0)
                 - gsl_vector_get(tree->shift, 0)),
              gsl_vector_get(tree->scale, 1)
              * (gsl_vector_get(&(p2.vector), 1)
                 - gsl_vector_get(tree->shift, 1)),
              r2);
    }
  gsl_vector *x0 = gsl_vector_alloc(tree->dim);
  double r2;
  calculate_hypersphere(tree, node, gdata, x0, &r2, tree->accel);
  fprintf(fcircles, "%g %g %g\n",
          gsl_vector_get(x0, 0),
          gsl_vector_get(x0, 1),
          sqrt(r2));
  gsl_vector_free(x0);
}

void
output_triangulation(simplex_tree *tree, gsl_matrix *data, gsl_vector *response,
                     char *lines_filename, char *points_filename,
                     char *circles_filename)
{
  flines = NULL;
  fcircles = NULL;
  if (lines_filename) flines = fopen(lines_filename, "w");
  if (circles_filename) fcircles = fopen(circles_filename, "w");

  if (points_filename)
    {
      FILE *fpoints = fopen(points_filename, "w");

      int i;
      for (i = 0; i < tree->n_points; i++)
        {
          fprintf(fpoints, "%g %g\n",
                  gsl_vector_get(tree->scale, 0)
                  * (gsl_matrix_get(data,
                                    gsl_permutation_get(tree->shuffle, i), 0)
                     - gsl_vector_get(tree->shift, 0)),
                  gsl_vector_get(tree->scale, 1)
                  * (gsl_matrix_get(data,
                                    gsl_permutation_get(tree->shuffle, i), 1)
                     - gsl_vector_get(tree->shift, 1)));
        }
      fclose(fpoints);
    }

  gdata = data;
  gresponse = response;
  struct node_list *seen = NULL;
  check_leaf_nodes(tree, tree->root, &seen, _output_triangulation);
  free_list(seen);

  if (flines) fclose(flines);
  if (fcircles) fclose(fcircles);
}

