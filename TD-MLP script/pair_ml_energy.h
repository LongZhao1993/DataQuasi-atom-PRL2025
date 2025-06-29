/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   see LLNL copyright notice at bottom of file
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(ml/energy,PairMLEnergy)

#else

#ifndef LMP_PAIR_ML_ENERGY_H
#define LMP_PAIR_ML_ENERGY_H

#include "pair.h"
#include <vector>
#include <dlib/svm.h>

using namespace std;
using namespace dlib;

namespace LAMMPS_NS {

/// Set this to 1 if you intend to use MEAM potentials with non-uniform spline knots.
/// Set this to 0 if you intend to use only MEAM potentials with spline knots on a uniform grid.
///
/// With SUPPORT_NON_GRID_SPLINES == 0, the code runs about 50% faster.

#define ML_ENERGY_SUPPORT_NON_GRID_SPLINES 0

class PairMLEnergy : public Pair
{
public:
        PairMLEnergy(class LAMMPS *);
        virtual ~PairMLEnergy();
        virtual void compute(int, int);
        void settings(int, char **);
        void coeff(int, char **);
        void init_style();
        void init_list(int, class NeighList *);
	    void grab(FILE *, int, double *);
        double init_one(int, int);
		
        int pack_forward_comm(int, int *, double *, int, int *);
        void unpack_forward_comm(int, int, double *);
        int pack_reverse_comm(int, int, double *);
        void unpack_reverse_comm(int, int *, double *);
        double memory_usage();
		

protected:
  char **elements;              // names of unique elements
  int *map;                     // mapping from atom types to elements
  double **rho_values;          // store manybody term
  int nelements,train_flag;      // # of unique elements
  int nfeature, ntarget,nrho, nlines,npairs,eta_num,nvector;
  double *Fvect_mpi_arr, *Target_mpi_arr, *eta_list,*eta2_list;
  double rho_cent,rho_scale; 
 
    typedef matrix<double,104,1> Fvect_Type;  
    Fvect_Type fvect_unit,fvect_unit_Eg;
    Fvect_Type norm_dev,norm_mean,fvect_dev,fvect_rho;	

    std::vector<Fvect_Type> fvect_arr;
    std::vector<double> target_arr;
  
    /// Helper data structure for potential routine.
    typedef radial_basis_kernel<Fvect_Type> kernel_type;
    krr_trainer<kernel_type> trainer;
    decision_function<kernel_type> final_pot_trainer;
    vector_normalizer<Fvect_Type> normalizer;	
    kernel_type kern;

		
        double zero_atom_energy;        // Shift embedding energy by this value to make it zero for a single atom in vacuum.
        double Pi_value = acos(-1.0);
		double electron_temperature_factor;
        double cutoff,cutoff2;              // The cutoff radius
		
	  /// Helper data structure for potential routine.
        struct MEAM2Body {
                int tag;
                double r;
                double fcut;
                double fcut_dev;				
                double del[3];
        };
		
        int nmax;                              // Size of temporary array.
        int maxNeighbors;                      // The last maximum number of neighbors a single atoms has.
        MEAM2Body* twoBodyInfo;                // Temporary array.

        void read_file(const char* filename);
        void read_param(const char* filename);		
        void allocate();
        double fun_cutoff(double, double);
        double fun_cutoff(double, double, double);
        double fun_cutoff_dev(double,double,double);
        double LJ_fpair(double );
        double LJ_Epair(double );
        double Hx(double );
        void   Data_Fitting();
        void   costheta_d(double,const double *, double,const double *, double, double *, double *);
		
  // vector functions, inline for efficiency
  inline double vec3_dot(const double *x,const double *y) {
    return x[0]*y[0] + x[1]*y[1] + x[2]*y[2];
  }
  inline void vec3_add(const double *x, const double *y, double *z) {
    z[0] = x[0]+y[0];  z[1] = x[1]+y[1];  z[2] = x[2]+y[2];
  }
  inline void vec3_scale(const double k, const double *x, double *y) {
    y[0] = k*x[0];  y[1] = k*x[1];  y[2] = k*x[2];
  }
  inline void vec3_scaleadd(const double k,const double *x,const double *y, double *z) {
    z[0] = k*x[0]+y[0];  z[1] = k*x[1]+y[1];  z[2] = k*x[2]+y[2];
  }
  
};

}

#endif
#endif

/* ----------------------------------------------------------------------
 * Spline-based Modified Embedded Atom method (MEAM) potential routine.
 *
 * Copyright (2011) Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by Alexander Stukowski (<alex@stukowski.com>).
 * LLNL-CODE-525797 All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2, dated June 1991.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE. See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * Our Preamble Notice
 * A. This notice is required to be provided under our contract with the
 * U.S. Department of Energy (DOE). This work was produced at the
 * Lawrence Livermore National Laboratory under Contract No.
 * DE-AC52-07NA27344 with the DOE.
 *
 * B. Neither the United States Government nor Lawrence Livermore National
 * Security, LLC nor any of their employees, makes any warranty, express or
 * implied, or assumes any liability or responsibility for the accuracy,
 * completeness, or usefulness of any information, apparatus, product, or
 * process disclosed, or represents that its use would not infringe
 * privately-owned rights.
 *
 * C. Also, reference herein to any specific commercial products, process,
 * or services by trade name, trademark, manufacturer or otherwise does not
 * necessarily constitute or imply its endorsement, recommendation, or
 * favoring by the United States Government or Lawrence Livermore National
 * Security, LLC. The views and opinions of authors expressed herein do not
 * necessarily state or reflect those of the United States Government or
 * Lawrence Livermore National Security, LLC, and shall not be used for
 * advertising or product endorsement purposes.
 *
 * See file 'pair_spline_meam.cpp' for history of changes.
------------------------------------------------------------------------- */
