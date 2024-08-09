#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include "datadef.h"

/* Modified slightly by D. Orchard (2010) from the classic code from: 

    Michael Griebel, Thomas Dornseifer, Tilman Neunhoeffer,
    Numerical Simulation in Fluid Dynamics,
    SIAM, 1998.

    http://people.sc.fsu.edu/~jburkardt/cpp_src/nast2d/nast2d.html

*/

/* Initialize the flag array, marking any obstacle cells and the edge cells
 * as boundaries. The cells adjacent to boundary cells have their relevant
 * flags set too.
 */
void init_flag(char **flag, int imax, int jmax, double delx, double dely, int *ibound, int shape, double xoffset, double yoffset, double rotation)
{
    int i, j;
    double mx, my, x, y, rad1;
    double cos_theta = cos(rotation);
    double sin_theta = sin(rotation);

    printf("Shape: %d\n", shape);

    if (shape == 0) {
        printf("Circular obstacle\n");
        /* Mask of a circular obstacle */
        mx = 20.0/41.0*jmax*dely + xoffset;
        my = mx + yoffset;
        rad1 = 5.0/41.0*jmax*dely;
        for (i=1;i<=imax;i++) {
            for (j=1;j<=jmax;j++) {
                x = (i-0.5)*delx - mx;
                y = (j-0.5)*dely - my;
                flag[i][j] = (x*x + y*y <= rad1*rad1)?C_B:C_F;
                // Apply rotation
                double x_rot = x * cos_theta - y * sin_theta;
                double y_rot = x * sin_theta + y * cos_theta;
                flag[i][j] = (x_rot*x_rot + y_rot*y_rot <= rad1*rad1)?C_B:C_F;
            }
        }
    }
    if (shape == 1) {
        printf("Triangular obstacle\n");
        /* Mask of a triangular obstacle */
        mx = 20.0/41.0*jmax*dely + xoffset;  // Center x-coordinate
        my = 20.0/41.0*jmax*dely + yoffset;  // Center y-coordinate
        double height = 10.0/41.0*jmax*dely;  // Height of the triangle
        double base = height;  // Assuming an isosceles right triangle for simplicity

        for (i = 1; i <= imax; i++) {
            for (j = 1; j <= jmax; j++) {
                x = (i - 0.5) * delx - mx;
                y = (j - 0.5) * dely - my;
                // Apply rotation
                double x_rot = x * cos_theta - y * sin_theta;
                double y_rot = x * sin_theta + y * cos_theta;

                // Check if the point (x_rot, y_rot) lies within the triangle
                if (x_rot >= 0 && y_rot >= 0 && x_rot + y_rot <= base) {
                    flag[i][j] = C_B;  // Inside the triangle
                } else {
                    flag[i][j] = C_F;  // Outside the triangle
                }
            }
        }
    }
    if (shape == 2) {
        printf("Square obstacle\n");
        /* Mask of a square obstacle */
        mx = 20.0/41.0*jmax*dely + xoffset;  // Center x-coordinate
        my = 20.0/41.0*jmax*dely + yoffset;  // Center y-coordinate
        double side = 10.0/41.0*jmax*dely;  // Side length of the square

        double half_side = side / 2.0;

        for (i = 1; i <= imax; i++) {
            for (j = 1; j <= jmax; j++) {
                x = (i - 0.5) * delx - mx;
                y = (j - 0.5) * dely - my;
                double x_rot = x * cos_theta - y * sin_theta;
                double y_rot = x * sin_theta + y * cos_theta;

                // Check if the point (x_rot, y_rot) lies within the square
                if (fabs(x_rot) <= half_side && fabs(y_rot) <= half_side) {
                    flag[i][j] = C_B;  // Inside the square
                } else {
                    flag[i][j] = C_F;  // Outside the square
                }
            }
        }
    }
    else {
        printf("Invalid shape\n");
    }

    // /* Mask of a triangular obstacle */
    // mx = 20.0/41.0*jmax*dely;  // Center x-coordinate
    // my = 20.0/41.0*jmax*dely;  // Center y-coordinate
    // double height = 10.0/41.0*jmax*dely;  // Height of the triangle
    // double base = height;  // Assuming an isosceles right triangle for simplicity

    // for (i = 1; i <= imax; i++) {
    //     for (j = 1; j <= jmax; j++) {
    //         x = (i - 0.5) * delx - mx;
    //         y = (j - 0.5) * dely - my;

    //         // Check if the point (x, y) lies within the triangle
    //         if (x >= 0 && y >= 0 && x + y <= base) {
    //             flag[i][j] = C_B;  // Inside the triangle
    //         } else {
    //             flag[i][j] = C_F;  // Outside the triangle
    //         }
    //     }
    // }

    // /* Mask of a square obstacle */
    // mx = 20.0/41.0*jmax*dely;  // Center x-coordinate
    // my = 20.0/41.0*jmax*dely;  // Center y-coordinate
    // double side = 10.0/41.0*jmax*dely;  // Side length of the square

    // double half_side = side / 2.0;

    // for (i = 1; i <= imax; i++) {
    //     for (j = 1; j <= jmax; j++) {
    //         x = (i - 0.5) * delx - mx;
    //         y = (j - 0.5) * dely - my;

    //         // Check if the point (x, y) lies within the square
    //         if (x >= -half_side && x <= half_side && y >= -half_side && y <= half_side) {
    //             flag[i][j] = C_B;  // Inside the square
    //         } else {
    //             flag[i][j] = C_F;  // Outside the square
    //         }
    //     }
    // }

    
    /* Mark the north & south boundary cells */
    for (i=0; i<=imax+1; i++) {
        flag[i][0]      = C_B;
        flag[i][jmax+1] = C_B;
    }
    /* Mark the east and west boundary cells */
    for (j=1; j<=jmax; j++) {
        flag[0][j]      = C_B;
        flag[imax+1][j] = C_B;
    }

    /* flags for boundary cells */
    *ibound = 0;
    for (i=1; i<=imax; i++) {
        for (j=1; j<=jmax; j++) {
            if (!(flag[i][j] & C_F)) {
                (*ibound)++;
                if (flag[i-1][j] & C_F) flag[i][j] |= B_W;
                if (flag[i+1][j] & C_F) flag[i][j] |= B_E;
                if (flag[i][j-1] & C_F) flag[i][j] |= B_S;
                if (flag[i][j+1] & C_F) flag[i][j] |= B_N;
            }
        }
    }
}
