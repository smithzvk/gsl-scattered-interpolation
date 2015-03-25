
struct node_list
{
  void *val;
  struct node_list *next;
};

void check_leaf_nodes(_simplex_tree *node, struct node_list **seen);
int in_list(_simplex_tree *node, struct node_list *list);
int cycle(struct node_list *list);
void free_list(struct node_list *list);

