#include <assert.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_permutation.h>

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
check_leaf_nodes(simplex_tree *tree, simplex_tree_node *node, struct node_list **seen)
{
  if (!node->leaf_p)
    {
      check_leaf_nodes(tree, node->links[0], seen);
      return;
    }
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

          /* Recurse but only if we haven't seen this node yet */
          assert(!cycle(*seen));
          if (!in_list(neighbor, *seen))
            {
              *seen = cons(neighbor, *seen);
              check_leaf_nodes(tree, neighbor, seen);
            }
        }
    }
}
