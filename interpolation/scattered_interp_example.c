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
  calculate_hypersphere(tree, 0, &(data.matrix),
                        &(center_vector.vector), &r2,
                        accel);

  /* Trivial leaf find */
  double point_vector[2] = {-88, 41};
  gsl_vector_view point = gsl_vector_view_array(point_vector, 2);
  simplex_index leaf = find_leaf(tree, &(data.matrix), &(point.vector), NULL);

  /* Test trivial interpolation */
  assert(0 == interp_point(tree, leaf, &(data.matrix), &(response.vector),
                           &(point.vector), accel));

  /* Now things get interesting */

  /* Inserting new points */
  gsl_vector_view new_point = gsl_matrix_row(&(data.matrix), 0);
  insert_point(tree, leaf, &(data.matrix), &(new_point.vector), accel);

  assert(!SIMP(leaf)->leaf_p);
  assert(0 == POINT(LINK(leaf, 0), 0));
  assert(-2 == POINT(LINK(leaf, 0), 1));
  assert(-3 == POINT(LINK(leaf, 0), 2));
  assert(0 == POINT(LINK(leaf, 1), 0));
  assert(-1 == POINT(LINK(leaf, 1), 1));
  assert(-3 == POINT(LINK(leaf, 1), 2));
  assert(0 == POINT(LINK(leaf, 2), 0));
  assert(-1 == POINT(LINK(leaf, 2), 1));
  assert(-2 == POINT(LINK(leaf, 2), 2));

  assert(1 == in_hypersphere(tree, 0, &(data.matrix), 0, accel));

  /* Finding the containing triangle */
  leaf = find_leaf(tree, &(data.matrix), &(point.vector), accel);
  assert(0 == POINT(leaf, 0));
  assert(-1 == POINT(leaf, 1));
  assert(-3 == POINT(leaf, 2));

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

  output_triangulation(tree, &(data.matrix), &(response.vector),
                       "/tmp/lines.dat", "/tmp/points.dat", "/tmp/circles.dat");

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

     ...and to draw with the circles (busy, but useful for debugging):

     gnuplot> set size ratio -1
     gnuplot> plot '/tmp/circles.dat' w circles, '/tmp/lines.dat' w lines

     ...and with points:

     gnuplot> plot '/tmp/points.dat' w points, \
                   '/tmp/circles.dat' w circles, \
                   '/tmp/lines.dat' w lines

  */

  gsl_matrix_free(grid);
  simplex_tree_free(tree);
  simplex_tree_accel_free(accel);
  return 0;
}
