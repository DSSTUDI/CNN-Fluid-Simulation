#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <errno.h>

#include "alloc.h"
#include "boundary.h"
#include "datadef.h"
#include "init.h"
#include "simulation.h"

/* Modified slightly by D. Orchard (2010) from the classic code from: 

    Michael Griebel, Thomas Dornseifer, Tilman Neunhoeffer,
    Numerical Simulation in Fluid Dynamics,
    SIAM, 1998.

    http://people.sc.fsu.edu/~jburkardt/cpp_src/nast2d/nast2d.html

*/

int main(int argc, char *argv[])
{

    int loops = 500;      /* Number of times to run the simulation */

    int verbose = 1;          /* Verbosity level */
    double xlength = 256.0;     /* Width of simulated domain */
    double ylength = 128.0;      /* Height of simulated domain */
    int imax = 256;           /* Number of cells horizontally */
    int jmax = 128;           /* Number of cells vertically */

    char *outname;
    int output = 0;
    int output_frequency = 0;

    double t_end = 500; //2.1       /* Simulation runtime */
    double del_t = 0.003;      /* Duration of each timestep */
    double tau = 0.5;          /* Safety factor for timestep control */

    int itermax = 100;        /* Maximum number of iterations in SOR */
    double eps = 0.001;        /* Stopping error threshold for SOR */
    double omega = 1.7;        /* Relaxation parameter for SOR */
    double gamma = 0.9;        /* Upwind differencing factor in PDE
                                 discretisation */

    double Re = 2.0;         /* Reynolds number */
    double ui = 1.0;           /* Initial X velocity */
    double vi = 0.0;           /* Initial Y velocity */

    double t, delx, dely;
    int  i, j, itersor = 0, ifluid = 0, ibound = 0;
    double res;
    double **u, **v, **p, **rhs, **f, **g, **shape_sdf;
    char  **flag;
    int init_case, iters = 0;
    int show_help = 0, show_usage = 0, show_version = 0;

    if (argc > 1) {
      output = 1;
      outname = argv[1];
      output_frequency = 1;
    }

    if (argc > 2) {
      output_frequency = atoi(argv[2]);
      float circle_x = atof(argv[2]);
      printf("circle_x: %f\n", circle_x);
    }

    delx = xlength/imax;
    dely = ylength/jmax;



    unsigned long checker = 0;
    double checker1 = 0.0;

    for (int l = 0; l < loops; l++) {

          /* Allocate arrays */
      u    = alloc_doublematrix(imax+2, jmax+2);
      v    = alloc_doublematrix(imax+2, jmax+2);
      f    = alloc_doublematrix(imax+2, jmax+2);
      g    = alloc_doublematrix(imax+2, jmax+2);
      p    = alloc_doublematrix(imax+2, jmax+2);
      rhs  = alloc_doublematrix(imax+2, jmax+2);
      shape_sdf = alloc_doublematrix(imax+2, jmax+2); 
      flag = alloc_charmatrix(imax+2, jmax+2);                 

      if (!u || !v || !f || !g || !p || !rhs || !flag) {
        fprintf(stderr, "Couldn't allocate memory for matrices.\n");
        return 1;
      }
      
      printf("Starting loop %d\n", l);
      
      // Set up initial values
      for (i=0;i<=imax+1;i++) {
          for (j=0;j<=jmax+1;j++) {
          checker += (i*jmax)+ j + 1;
        checker1 += (i*jmax) + j + 1.0;
              u[i][j] = ui;
              v[i][j] = vi;
              p[i][j] = 0.0;
          }
      }

      int shape = rand() % 3; /* Shape of the obstacle 0 = circle, 1 = triangle, 2 = square */

      // printf("Shape number: %d\n", shape);

      double xoffset = -20.0 + (220.0 * ((double)rand() / RAND_MAX));
      double yoffset = -40.0 + (80 * ((double)rand() / RAND_MAX));
      double rotation = 0 + ((double)rand() / RAND_MAX) * (6.2 - 0);

      init_flag(flag, imax, jmax, delx, dely, &ibound, shape, xoffset, yoffset, rotation);
      apply_boundary_conditions(u, v, flag, imax, jmax, ui, vi);
      
      // Main loop

      iters = 0;
      for (t = 0.0; t < t_end; t += del_t, iters++) {
          set_timestep_interval(&del_t, imax, jmax, delx, dely, u, v, Re, tau);

          ifluid = (imax * jmax) - ibound;

          compute_tentative_velocity(u, v, f, g, flag, imax, jmax,
              del_t, delx, dely, gamma, Re);

          compute_rhs(f, g, rhs, flag, imax, jmax, del_t, delx, dely);

          if (ifluid > 0) {
              itersor = poisson(p, rhs, flag, imax, jmax, delx, dely,
                          eps, itermax, omega, &res, ifluid);
          } else {
              itersor = 0;
          }

          // printf("%d t:%g, del_t:%g, SOR iters:%3d, res:%e, bcells:%d\n", iters, t+del_t, del_t, itersor, res, ibound);

    
          update_velocity(u, v, f, g, p, flag, imax, jmax, del_t, delx, dely);

          apply_boundary_conditions(u, v, flag, imax, jmax, ui, vi);

          if (output && (iters % output_frequency == 0)) {
            write_ppm(u, v, p, flag, imax, jmax, xlength, ylength, outname,
            iters, output_frequency);
          }
      }

      for (i = 0; i <= imax + 1; i++) {
        for (j = 0; j <= jmax + 1; j++) {
          if (flag[i][j] != C_F) {
            shape_sdf[i][j] = -1; 
          }
          else {
            shape_sdf[i][j] = 0;
          }
        }
      }
      

      // Save vertical velocity to csv
      char filename[40];
      sprintf(filename, "vertical/v_%d.csv", l);
      FILE *csv_file_v = fopen(filename, "w");
      if (csv_file_v == NULL) {
        fprintf(stderr, "Failed to open v.csv for writing.\n");
        return 1;
      }

      for (i = 0; i <= imax + 1; i++) {
        for (j = 0; j <= jmax + 1; j++) {
          fprintf(csv_file_v, "%f,", v[i][j]);
        }
        fprintf(csv_file_v, "\n");
      }

      fclose(csv_file_v);

      // Save horizontal velocity to csv
      sprintf(filename, "horizontal/u_%d.csv", l);
      FILE *csv_file_u = fopen(filename, "w");
      if (csv_file_u == NULL) {
        fprintf(stderr, "Failed to open u.csv for writing.\n");
        return 1;
      }

      for (i = 0; i <= imax + 1; i++) {
        for (j = 0; j <= jmax + 1; j++) {
          fprintf(csv_file_u, "%f,", u[i][j]);
        }
        fprintf(csv_file_u, "\n");
      }

      fclose(csv_file_u);

      // Save shape to csv
      sprintf(filename, "shape/sdf_%d.csv", l);
      FILE *sdf_file = fopen(filename, "w");
      if (sdf_file == NULL) {
        fprintf(stderr, "Failed to open u.csv for writing.\n");
        return 1;
      }

      for (i = 0; i <= imax + 1; i++) {
        for (j = 0; j <= jmax + 1; j++) {
          fprintf(sdf_file, "%f,", shape_sdf[i][j]);
        }
        fprintf(sdf_file, "\n");
      }

      fclose(sdf_file);



      free_matrix(u);
      free_matrix(v);
      free_matrix(f);
      free_matrix(g);
      free_matrix(p);
      free_matrix(rhs);
      free_matrix(flag);


    }
      return 0;
}

// Used for comparing computations when debugging other implementations

unsigned int simplest_checksum_char(char** in, int imax, int jmax) {
  unsigned int checksum = 0;
  int i;
  int j;
  for (i=0; i<(imax+2); i++){
    for (j=0; j<(jmax+2); j++){
      checksum+=in[i][j]*(i);
    }
  }
  return checksum;
}

double simplest_checksum(double** in, int imax, int jmax) {
  double checksum = 0.0;
  int i;
  int j;
  for (i=0; i<(imax+2); i++){
    for (j=0; j<(jmax+2); j++){
      checksum+=in[i][j]*((double)(i*jmax)+j);
    }
  }
  return checksum;
}
