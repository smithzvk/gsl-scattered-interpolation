#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <gsl/gsl_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_interpsc.h>
#include <gsl/gsl_rng.h>
#include "linear_simplex.h"
#include "linear_simplex_integrity_check.h"

/* Some cool things to do:

1. Interpolate temperature data.  Make gridded data that you can plot using
   gnuplot

2. Interpolate temperature data with time as a dimension.  Use this to plot the
   temperature along a curve in time and space, like plot the temperature
   experienced by a car as it drives along a path.

*/

FILE *triangle_plot;
gsl_matrix *gdata;

void output_lines(simplex_tree *tree, simplex_tree_node *node)
{
  int i;
  for (i = 0; i < tree->dim + 1; i++)
    {
      int i1 = node->points[i];
      int i2 = node->points[(i+1)%(tree->dim + 1)];
      if (i1 < 0 || i2 < 0) continue;
      gsl_vector_view p1
        = gsl_matrix_row(gdata, gsl_permutation_get(tree->shuffle, i1));
      gsl_vector_view p2
        = gsl_matrix_row(gdata, gsl_permutation_get(tree->shuffle, i2));
      fprintf(triangle_plot,
              "%g %g 0\n%g %g 0\n\n",
              gsl_vector_get(&(p1.vector), 0),
              gsl_vector_get(&(p1.vector), 1),
              gsl_vector_get(&(p2.vector), 0),
              gsl_vector_get(&(p2.vector), 1));
    }
}

int
main()
{
  /* Trivial allocating and deallocating a triangulation */
  simplex_tree *tree = simplex_tree_alloc(2, 10);
  simplex_tree_free(tree);

  /* Loading the data */
#include "weather_data.c"
  gsl_matrix_view all_data = gsl_matrix_view_array(data_vec, 50, 3);
  gsl_vector_view response = gsl_matrix_column(&(all_data.matrix), 2);
  gsl_matrix_view data = gsl_matrix_submatrix(&(all_data.matrix), 0, 0, 50, 2);

  double sphere_center[] = {0, 0};
  gsl_vector_view center_vector = gsl_vector_view_array(sphere_center, 2);
  double r2;
  simplex_tree_accel *accel = simplex_tree_accel_alloc(2);

  /* Test calculate hypersphere */
  tree = simplex_tree_alloc(2, 50);
  simplex_tree_init(tree, NULL, 0, NULL);
  calculate_hypersphere(tree, tree->root, &(data.matrix),
                        &(center_vector.vector), &r2,
                        accel);

  /* Trivial leaf find */
  double point_vector[2] = {-88, 41};
  gsl_vector_view point = gsl_vector_view_array(point_vector, 2);
  simplex_tree_node *leaf = find_leaf(tree, &(data.matrix), &(point.vector), NULL);

  /* Test trivial interpolation */
  assert(0 == interp_point(tree, leaf, &(data.matrix), &(response.vector),
                           &(point.vector), accel));

  /* Now things get interesting */

  /* Inserting new points */
  gsl_vector_view new_point = gsl_matrix_row(&(data.matrix), 0);
  insert_point(tree, leaf, &(data.matrix), &(new_point.vector), accel);

  assert(!leaf->leaf_p);
  assert(3 == leaf->n_links);
  assert(0 == leaf->links[0]->points[0]);
  assert(-2 == leaf->links[0]->points[1]);
  assert(-3 == leaf->links[0]->points[2]);
  assert(0 == leaf->links[1]->points[0]);
  assert(-1 == leaf->links[1]->points[1]);
  assert(-3 == leaf->links[1]->points[2]);
  assert(0 == leaf->links[2]->points[0]);
  assert(-1 == leaf->links[2]->points[1]);
  assert(-2 == leaf->links[2]->points[2]);

  assert(1 == in_hypersphere(tree, tree->root, &(data.matrix), 0, accel));

  /* Finding the containing triangle */
  leaf = find_leaf(tree, &(data.matrix), &(point.vector), accel);
  assert(0 == leaf->points[0]);
  assert(-1 == leaf->points[1]);
  assert(-3 == leaf->points[2]);

  /* Building simplex trees */
  int i;
  for (i = 1; i < 50; i++)
    {
      new_point = gsl_matrix_row(&(data.matrix), i);
      leaf = find_leaf(tree, &(data.matrix), &(new_point.vector), accel);
      insert_point(tree, leaf, &(data.matrix), &(new_point.vector), accel);
    }

  leaf = find_leaf(tree, &(data.matrix), &(point.vector), accel);
  double res = interp_point(tree, leaf, &(data.matrix), &(response.vector),
                            &(point.vector), accel);

  simplex_tree_free(tree);

  tree = simplex_tree_alloc(2, 50);

  gsl_rng_env_setup();
  gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);
  simplex_tree_init(tree, &(data.matrix), 0, rng);
  gsl_rng_free(rng);

  leaf = find_leaf(tree, &(data.matrix), &(point.vector), accel);
  res = interp_point(tree, leaf, &(data.matrix), &(response.vector),
                     &(point.vector), accel);

  /* Gridding data */
  double min[] = {1000000, 1000000};
  double max[] = {-1000000, -1000000};
  int j;
  for (i = 0; i < 50; i++)
    {
      for (j = 0; j < data.matrix.size2; j++)
        {
          int val = gsl_matrix_get(&(data.matrix), i, j);
          if (min[j] > val)
            min[j] = val;
          if (max[j] < val)
            max[j] = val;
        }
    }

  int n_grid = 100;

  gsl_matrix *grid = gsl_matrix_alloc(n_grid, n_grid);
  double x, y;
  double overscan = .5;
  for (i = 0; i < n_grid; i++)
    {
      double xrange = (max[0]-min[0]);
      double xstep = xrange/n_grid;
      x = min[0] - overscan*xrange + (1 + 3*overscan)*xstep * i;
      gsl_vector_set(&(point.vector), 0, x);
      for (j = 0; j < n_grid; j++)
        {
          double yrange = (max[1]-min[1]);
          double ystep = yrange/n_grid;
          y = min[1] - overscan*yrange + (1 + 3*overscan)*ystep * j;

          gsl_vector_set(&(point.vector), 1, y);
          leaf = find_leaf(tree, &(data.matrix), &(point.vector), accel);
          double res = interp_point(tree, leaf, &(data.matrix),
                                    &(response.vector), &(point.vector), accel);
          gsl_matrix_set(grid, i, j, res);
        }
    }

  triangle_plot = fopen("/tmp/lines.dat", "w");
  gdata = &(data.matrix);
  struct node_list *seen = NULL;
  check_leaf_nodes(tree, tree->root, &seen, output_lines);
  free_list(seen);
  fclose(triangle_plot);

  FILE *plot = fopen("/tmp/plot.dat", "w");
  for (i = 0; i < n_grid; i++)
    {
      gsl_vector_view row = gsl_matrix_row(grid, i);
      gsl_vector_fprintf(plot, &(row.vector), "%g");
      fprintf(plot, "\n");
    }
  fclose(plot);

  gsl_matrix_free(grid);
  simplex_tree_free(tree);
  simplex_tree_accel_free(accel);
  return 0;
}
