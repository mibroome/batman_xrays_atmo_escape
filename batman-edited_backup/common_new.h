/* The batman package: fast computation of exoplanet transit light curves
 * Copyright (C) 2015 Laura Kreidberg
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Minor edits for use with limb-brighted, X-ray transits distributed under the same terms
 * Copyright (C) 2024 George King
 */


#if defined (_OPENACC) && defined(__PGI)
#  include <accelmath.h>
#else
#  include <math.h>
#endif

#if defined (_OPENMP) && !defined(_OPENACC)
#  include <omp.h>
#endif

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#include "numpy/arrayobject.h"

#include <stdio.h>
#include <stdlib.h>

// Global tau array and its length
double *tau = NULL;
int tau_len = 0;

// Function to read tau values from file (variable length)
static inline int read_tau_from_file(const char *filename) {
	FILE *file = fopen(filename, "r");
	if (!file) return -1;

	// First, count the number of values
	int count = 0;
	double temp;
	while (fscanf(file, "%lf", &temp) == 1) {
		count++;
	}
	rewind(file);

	// Allocate tau array
	tau = (double *)malloc(count * sizeof(double));
	if (!tau) {
		fclose(file);
		return -3; // Memory allocation failed
	}
	tau_len = count;

	// Read values into tau
	for (int i = 0; i < count; i++) {
		if (fscanf(file, "%lf", &tau[i]) != 1) {
			free(tau);
			tau = NULL;
			tau_len = 0;
			fclose(file);
			return -2; // Not enough values
		}
	}
	fclose(file);
	return 0;
}

/* Must be defined in the C file that includes this header. */
inline double intensity(double x, double intVal[10001]);

// static inline double tau(double ir) {
// 	double tau0 = 1; // central optical depth
// 	return tau0/(pow(ir,2)+1);
// 	// return tau0 * exp(-r / r0);
// }

double overlapping_area(double d, double x, double r)
{
    if (d >= x + r) return 0.0;
    if (d <= fabs(x - r)) return M_PI * fmin(x, r) * fmin(x, r);

    double arg1 = (d*d + x*x - r*r)/(2.0*d*x);
    double arg2 = (d*d + r*r - x*x)/(2.0*d*r);
    double arg3 = (-d + x + r)*(d + x - r)*(d - x + r)*(d + x + r);

    return x*x*acos(arg1) + r*r*acos(arg2) - 0.5*sqrt(arg3);
}


double area(double d, double x, double R)
{
	/* Returns area of overlapping circles with radii x and R; separated by a distance d */

	// double arg1 = (d*d + x*x - R*R)/(2.*d*x);
	// double arg2 = (d*d + R*R - x*x)/(2.*d*R);
	// double arg3 = MAX((-d + x + R)*(d + x - R)*(d - x + R)*(d + x + R), 0.);

	int Nsteps = tau_len > 0 ? tau_len : 100; // radial steps
	double dr = R / Nsteps;
 
	if(x <= R - d) return M_PI*x*x;						//planet completely overlaps stellar circle
	// else if(x >= R + d) return M_PI*R*R;						//stellar circle completely overlaps planet
	else if(x >= R + d){
		double blocked_flux = 0.0;
		for (int ir = 0; ir < Nsteps; ++ir) {
			double r_n = dr * (ir + 0.5);
			double transmission = 1.0;
			if (tau && ir < tau_len) transmission = 1.0 - exp(-tau[ir]);
			if (transmission < 0) transmission = 0.0;
			if (transmission > 1.0) transmission = 1.0;
			blocked_flux += transmission * 2.0 * M_PI * r_n * dr;
		}
		return blocked_flux;
	}
	else{
		double blocked_flux = 0.0;

		for (int ir = 0; ir < Nsteps; ++ir) {
			double r1 = dr * ir;
			double r2 = dr * (ir + 1);

			double A1 = overlapping_area(d, x, r1);
			double A2 = overlapping_area(d, x, r2);

			double transmission = 1.0;
			if (tau && ir < tau_len) transmission = 1.0 - exp(-tau[ir]);
			if (transmission < 0) transmission = 0.0;
			if (transmission > 1.0) transmission = 1.0;

			blocked_flux += transmission * (A2 - A1);
		}
		return blocked_flux;
	}
	
	if (tau) {
    free(tau);
    tau = NULL;
    tau_len = 0;
	}
}
	// else return (x*x*acos(arg1) + R*R*acos(arg2) - 0.5*sqrt(arg3));			//partial overlap


	// else if(x >= R + d) return M_PI*R*R;						//stellar circle completely overlaps planet
	// // else return x*x*acos(arg1) + R*R*acos(arg2) - 0.5*sqrt(arg3);			//partial overlap
	// else { 										//partial overlap
	// 	double blocked_flux = 0.0;
	// 	for (int ir = 0; ir < Nsteps; ++ir) {
	// 		double r_n = dr * (ir + 0.5);
	// 		double transmission = 1.0;// - exp(-tau(ir));
	// 		if (abs(d+x)>r_n){
	// 			double arg = (d*d + r_n*r_n - x*x)/(2.*d*r_n); //law of cosines
	// 			FILE *fp = fopen("test_new", "a");
	// 			if (fp != NULL) {
	// 				fprintf(fp, "%f,%f,%f,%f,%f, %f\n", x, d, R, r_n, arg, transmission);
	// 				fclose(fp);
	// 			}	
	// 			double theta_n = acos(arg);
	// 			double arc = r_n * 2.0 * theta_n; //2pi cancels out
	// 			blocked_flux += transmission * arc * dr;
	// 		}
	// 		else{
	// 			blocked_flux += transmission * 2.0 * M_PI * r_n * dr;
	// 		}
	// 	}
	// 	return blocked_flux;
	// }


void calc_limb_darkening(double* f_array, double* d_array, int N, double rprs, double fac, int nthreads, double* intVals[10001])
{
	/*
		This function takes an array of sky distances (d_array) of length N, computes stellar intensity by calling intensity with
		intensity_args, and puts the results in f_array.  To use this function, include this file in a .c file and implement
		the intensity function within that .c file.

		The proper way of implementing this function is to accept a pointer to the intensity function.  Unfortunately, few
		compilers that implement OpenACC support function pointers, so this approach is not yet possible.
	*/

	#if defined (_OPENMP) && !defined(_OPENACC)
	omp_set_num_threads(nthreads);	//specifies number of threads (if OpenMP is supported)
	#endif

	#if defined (_OPENACC)
	#pragma acc parallel loop copyout(f_array[:N]) present(intensity_args)
	#elif defined (_OPENMP)
	#pragma omp parallel for
	#endif
	for(int i = 0; i < N; i++)
	{
		double d = d_array[i];
		double x_in = MAX(d - rprs, 0.);					//lower bound for integration
		double x_out = MIN(d + rprs, 1.0);					//upper bound for integration

		if(x_in >= 1.) f_array[i] = 1.0;					//flux = 1. if the planet is not transiting
		else if(x_out - x_in < 1.e-7) f_array[i] = 1.0;				//pathological case
		else
		{
			double delta = 0.;						//variable to store the integrated intensity, \int I dA
			double x = x_in;						//starting radius for integration
			double dx = 0.1*fac; 					//initial step size

			x += dx;						//first step

			double A_i = 0.;						//initial area

			while(x < x_out)
			{
				double A_f = area(d, x, rprs);				//calculates area of overlapping circles
				double I = intensity(x - dx/2., intVals); 	//intensity at the midpoint
				delta += (A_f - A_i)*I;				//increase in transit depth for this integration step
				dx = 0.1*fac;  				//updating step size
				x = x + dx;					//stepping to next element
				A_i = A_f;					//storing area
			}
			dx = x_out - x + dx;  					//calculating change in radius for last step  FIXME
			x = x_out;						//final radius for integration
			double A_f = area(d, x, rprs);					//area for last integration step
			double I = intensity(x - dx/2., intVals); 		//intensity at the midpoint
			delta += (A_f - A_i)*I;					//increase in transit depth for this integration step

			f_array[i] = 1.0 - delta;	//flux equals 1 - \int I dA
		}
	}
}
