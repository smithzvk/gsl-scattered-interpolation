
struct node_list
{
  void *val;
  struct node_list *next;
};

void check_leaf_nodes(simplex_tree *tree, simplex_tree_node *node,
                      struct node_list **seen,
                      void (*fn)(simplex_tree *, simplex_tree_node *));
int in_list(simplex_tree_node *node, struct node_list *list);
int cycle(struct node_list *list);
void free_list(struct node_list *list);

