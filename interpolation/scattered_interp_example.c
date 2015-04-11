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
FILE *circle_plot;
gsl_matrix *gdata;
gsl_vector *gresponse;

void output_lines_and_circles(simplex_tree *tree, simplex_tree_node *node)
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
          r1 = gsl_vector_get(gresponse, gsl_permutation_get(tree->shuffle, i1));
        }

      if (i2 < 0)
        {
          p2 = gsl_matrix_row(tree->seed_points, -i2 - 1);
          r2 = 0;
        }
      else
        {
          p2 = gsl_matrix_row(gdata, gsl_permutation_get(tree->shuffle, i2));
          r2 = gsl_vector_get(gresponse, gsl_permutation_get(tree->shuffle, i2));
        }

      fprintf(triangle_plot,
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
  fprintf(circle_plot, "%g %g %g\n",
          gsl_vector_get(x0, 0),
          gsl_vector_get(x0, 1),
          sqrt(r2));
  gsl_vector_free(x0);
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
  simplex_tree_init(tree, NULL, NULL, NULL, SIMPLEX_TREE_NOSTANDARDIZE, NULL);
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

  double min[] = {-89.6763, 40.9479};
  double max[] = {-86.303, 43.20};

  gsl_rng_env_setup();
  gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);
  simplex_tree_init(tree, &(data.matrix), NULL, NULL, 0, rng);
  gsl_rng_free(rng);

  leaf = find_leaf(tree, &(data.matrix), &(point.vector), accel);
  res = interp_point(tree, leaf, &(data.matrix), &(response.vector),
                     &(point.vector), accel);

  /* Gridding data */
  int n_grid = 100;

  gsl_matrix *grid = gsl_matrix_alloc(n_grid, n_grid);
  double x, y;
  double xrange = (max[0]-min[0]);
  double xstep = xrange/n_grid;
  double yrange = (max[1]-min[1]);
  double ystep = yrange/n_grid;
  for (i = 0; i < n_grid; i++)
    {
      x = min[0] + xstep * i;
      gsl_vector_set(&(point.vector), 0, x);
      int j;
      for (j = 0; j < n_grid; j++)
        {
          y = min[1] + ystep * j;

          gsl_vector_set(&(point.vector), 1, y);
          leaf = find_leaf(tree, &(data.matrix), &(point.vector), accel);
          double res = interp_point(tree, leaf, &(data.matrix),
                                    &(response.vector), &(point.vector), accel);
          gsl_matrix_set(grid, i, j, res);
        }
    }

  triangle_plot = fopen("/tmp/lines.dat", "w");
  circle_plot = fopen("/tmp/circles.dat", "w");
  gdata = &(data.matrix);
  gresponse = &(response.vector);
  struct node_list *seen = NULL;
  check_leaf_nodes(tree, tree->root, &seen, output_lines_and_circles);
  free_list(seen);
  fclose(triangle_plot);
  fclose(circle_plot);

  FILE *plot = fopen("/tmp/plot.dat", "w");
  for (i = 0; i < n_grid; i++)
    {
      int j;
      for (j = 0; j < n_grid; j++)
        {
          fprintf(plot, "%g %g %g\n",
                  min[0] + xstep * i,
                  min[1] + ystep * j,
                  gsl_matrix_get(grid, i, j));
        }
      fprintf(plot, "\n");
    }
  fclose(plot);

  /* Plot the data and the triangulation with Gnuplot using:

     gnuplot> set size ratio -1
     gnuplot> set view map
     gnuplot> unset key
     gnuplot> splot '/tmp/plot.dat' with pm3d, '/tmp/lines.dat' w lines

     ...or, if you prefer in 3d:

     gnuplot> unset view; replot

     To plot just the triangulation:

     gnuplot> plot '/tmp/lines.dat' w lines

     ...and to draw with the circles:

     gnuplot> set size ratio -1
     gnuplot> plot '/tmp/circles.dat' w circles, '/tmp/lines.dat' w lines

  */

  gsl_matrix_free(grid);
  simplex_tree_free(tree);
  simplex_tree_accel_free(accel);
  return 0;
}
