#include <gsl/gsl_test.h>
#include <gsl/gsl_ieee_utils.h>
#include <gsl/gsl_math.h>
#include "gsl_cblas.h"

void
test_amax () {
const double flteps = 1e-4, dbleps = 1e-6;
  {
   int N = 1;
   float X[] = { -0.388 };
   int incX = -1;
   float expected = 0;
   int k;
   k = cblas_isamax(N, X, incX);
   gsl_test_int(k, expected, "samax(case 52)");
  };


  {
   int N = 1;
   double X[] = { 0.247 };
   int incX = -1;
   double expected = 0;
   int k;
   k = cblas_idamax(N, X, incX);
   gsl_test_int(k, expected, "damax(case 53)");
  };


  {
   int N = 1;
   float X[] = { 0.704, 0.665 };
   int incX = -1;
   float expected = 0;
   int k;
   k = cblas_icamax(N, X, incX);
   gsl_test_int(k, expected, "camax(case 54)");
  };


  {
   int N = 1;
   double X[] = { -0.599, -0.758 };
   int incX = -1;
   double expected = 0;
   int k;
   k = cblas_izamax(N, X, incX);
   gsl_test_int(k, expected, "zamax(case 55)");
  };


  {
   int N = 2;
   float X[] = { 0.909, 0.037 };
   int incX = 1;
   float expected = 0;
   int k;
   k = cblas_isamax(N, X, incX);
   gsl_test_int(k, expected, "samax(case 56)");
  };


  {
   int N = 2;
   double X[] = { 0.271, -0.426 };
   int incX = 1;
   double expected = 1;
   int k;
   k = cblas_idamax(N, X, incX);
   gsl_test_int(k, expected, "damax(case 57)");
  };


  {
   int N = 2;
   float X[] = { -0.648, 0.317, 0.62, 0.392 };
   int incX = 1;
   float expected = 1;
   int k;
   k = cblas_icamax(N, X, incX);
   gsl_test_int(k, expected, "camax(case 58)");
  };


  {
   int N = 2;
   double X[] = { -0.789, 0.352, 0.562, 0.697 };
   int incX = 1;
   double expected = 1;
   int k;
   k = cblas_izamax(N, X, incX);
   gsl_test_int(k, expected, "zamax(case 59)");
  };


  {
   int N = 2;
   float X[] = { 0.487, 0.918 };
   int incX = -1;
   float expected = 0;
   int k;
   k = cblas_isamax(N, X, incX);
   gsl_test_int(k, expected, "samax(case 60)");
  };


  {
   int N = 2;
   double X[] = { 0.537, 0.826 };
   int incX = -1;
   double expected = 0;
   int k;
   k = cblas_idamax(N, X, incX);
   gsl_test_int(k, expected, "damax(case 61)");
  };


  {
   int N = 2;
   float X[] = { 0.993, 0.172, -0.825, 0.873 };
   int incX = -1;
   float expected = 0;
   int k;
   k = cblas_icamax(N, X, incX);
   gsl_test_int(k, expected, "camax(case 62)");
  };


  {
   int N = 2;
   double X[] = { 0.913, -0.436, -0.134, 0.129 };
   int incX = -1;
   double expected = 0;
   int k;
   k = cblas_izamax(N, X, incX);
   gsl_test_int(k, expected, "zamax(case 63)");
  };


}