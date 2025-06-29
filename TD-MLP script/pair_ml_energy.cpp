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
   Contributing author: Hongxiang Zong (XJTU& UoE), zonghust@mail.xjtu.edu.cn
   The University of Edinburgh
------------------------------------------------------------------------- */

#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "pair_ml_energy.h"
#include "atom.h"
#include "force.h"
#include "update.h"
#include "comm.h"
#include "memory.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "memory.h"
#include "error.h"
#include <iostream>
#include <vector>

#include <dlib/svm.h>

using namespace std;
using namespace dlib;
using namespace LAMMPS_NS;
#define MAXLINE 1024
/* ---------------------------------------------------------------------- */

PairMLEnergy::PairMLEnergy(LAMMPS *lmp) : Pair(lmp)
{
  nfeature = 1;
  ntarget = 1;
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 0;

  nelements = 0;
  elements = NULL;
  
  eta_list = NULL;
  eta2_list = NULL;  
  rho_values = NULL;
  twoBodyInfo = NULL;

  nmax = 0;
  eta_num = 0;
  maxNeighbors = 0;
  zero_atom_energy = 0.0;
  
  electron_temperature_factor = 0.0;

  comm_forward = 24;
  comm_reverse = 0;
}

/* ---------------------------------------------------------------------- */

PairMLEnergy::~PairMLEnergy()
{
  if (elements)
    for (int i = 0; i < nelements; i++) delete [] elements[i];
  delete [] elements;

  memory->destroy(Fvect_mpi_arr);
  memory->destroy(Target_mpi_arr);

  fvect_arr.clear();
  fvect_arr.shrink_to_fit();
  
  target_arr.clear();
  target_arr.shrink_to_fit();
  
  delete[] twoBodyInfo;
  memory->destroy(rho_values);  
  
  if(allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    delete [] map;
  }

  memory->destroy(eta_list); 
  memory->destroy(eta2_list);  
}

/* ---------------------------------------------------------------------- */

void PairMLEnergy::compute(int eflag, int vflag)
{
  if (eflag || vflag) ev_setup(eflag, vflag);
  else evflag = vflag_fdotr =
         eflag_global = vflag_global = eflag_atom = vflag_atom = 0;

  double cutforcesq = cutoff*cutoff;

  /// Grow per-atom array if necessary
  if (atom->nmax > nmax) {
	memory->destroy(rho_values);
    nmax = atom->nmax;
    memory->create(rho_values,nmax,24,"pair:rho_values");
  }

  double** const x = atom->x;
  double** v = atom->v;
  int* type = atom->type;
  double* mass = atom->mass;
  double** forces = atom->f;
  int nlocal = atom->nlocal;
  bool newton_pair = force->newton_pair;

  int inum_full = listfull->inum;
  int* ilist_full = listfull->ilist;
  int* numneigh_full = listfull->numneigh;
  int** firstneigh_full = listfull->firstneigh;

  int newMaxNeighbors = 0;
  for(int ii = 0; ii < inum_full; ii++) {
    int jnum = numneigh_full[ilist_full[ii]];
    if(jnum > newMaxNeighbors) newMaxNeighbors = jnum;
  }

  /// Allocate array for temporary bond info

  if(newMaxNeighbors > maxNeighbors) {
    maxNeighbors = newMaxNeighbors;
    delete[] twoBodyInfo;
    twoBodyInfo = new MEAM2Body[maxNeighbors];
  }

///double mu_list[8] ={2.562,3.315,4.688,5.324,6.254,6.797,7.986,8.213};
///double mu_list[8] ={4.595,6.594,8.037,9.190,10.396,11.189,12.188,13.785};
double mu_list[8] ={3.5266,4.2446,5.0049,5.6065,6.1239,6.5674,7.0426,7.5662};
double lamda_list[8] ={1.0,1.28,1.667,2.05,2.35,2.662,2.986,3.315};                   


double Re = cutoff*0.5;
double Re_sq = Re*Re;
double atom_temperature_factor;
double mv2toeV = 0.000103643; // unit: eV
double Boltzman_constant = 8.61734e-05; //unit: eV/K

double t=0;

 for(int ii=0;ii<inum_full;ii++){
	/// calculate atomic temperature //	
	int i = ilist_full[ii];
	t += 0.5*(v[i][0]*v[i][0]+v[i][1]*v[i][1]+v[i][2]*v[i][2])*mass[type[i]];	 
 }
 
 	atom_temperature_factor = 0.02*t*mv2toeV/(3.0*Boltzman_constant*inum_full);


 for(int ii = 0; ii < inum_full; ii++) {
    int i = ilist_full[ii];

/// calculate atomic temperature //	
	/*double t = 0.0;
	t += 0.5*(v[i][0]*v[i][0]+v[i][1]*v[i][1]+v[i][2]*v[i][2])*mass[type[i]];
	temperature_factor = 0.02*t*mv2toeV/(3.0*Boltzman_constant);*/
///Debug
    /*if(i<1)
    printf("temperature_factor==%lg\n",temperature_factor);*/

   for(int k=0;k<24;k++)
	   rho_values[i][k] = 0.0;	
	
    double xtmp = x[i][0];
    double ytmp = x[i][1];
    double ztmp = x[i][2];
    int* jlist = firstneigh_full[i];
    int jnum = numneigh_full[i];
	

    int numBonds = 0;
    double fagl = 0;
    double  evdwl = 0.0;	
    MEAM2Body* nextTwoBodyInfo = twoBodyInfo;   
    double PotEg = 0.0;

    for(int k=0;k<eta_num;k++)
	fvect_unit_Eg(k) = 0.0;
  
    	
    for(int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj];
      j &= NEIGHMASK;

      double jdelx = x[j][0] - xtmp;
      double jdely = x[j][1] - ytmp;
      double jdelz = x[j][2] - ztmp;
      double rij_sq = jdelx*jdelx + jdely*jdely + jdelz*jdelz;
 
      if(rij_sq < cutforcesq) {
        double rij = sqrt(rij_sq);		

       if(rij <2.0)
         {
           PotEg += LJ_Epair(rij);
           continue;
             }

        double fc_rij= fun_cutoff(rij,cutoff);
        for(int k=0;k<8;k++){
	    double lastp = exp(-1.0*pow(rij/eta_list[k],2))*fc_rij;
			rho_values[i][k] +=	lastp;

            fvect_unit_Eg(k) += lastp;
            double fc_exp = exp(-1.0*pow(rij/cutoff,2))*fc_rij;
            fvect_unit_Eg(k+8) += Hx(mu_list[k]-rij)*pow(mu_list[k]-rij,3)*fc_exp;
            fvect_unit_Eg(k+80) += 1.0*pow(atom_temperature_factor,2)*fc_rij/(1+exp(-1.0*(rij)/eta_list[k]));
			lastp = fc_rij*pow((rij-mu_list[k])/1.80, 2)*pow((1.0-1.0*(rij-mu_list[k])/1.80),2)*Hx(mu_list[k]-rij)*Hx(rij-mu_list[k]+1.80);
	        fvect_unit_Eg(k+88)+=  1.0*pow(electron_temperature_factor,2)*lastp;
            //rho_values[i][k+16] += lastp;			
            //rho_values[i][k] +=	(1+0.5*cos(k*rij))*fc_rij*exp(-1.0*rij/eta_list[k]);
            rho_values[i][k+8] += exp(-1.0*pow((rij-mu_list[k])/Re,2))*fc_rij;			
           // rho_values[i][k+16] += fc_rij*sin((2*k+1)*rij)*exp(-1.0*pow(rij/3.315,2));		
               }
        if(rij > 6.0) continue;
        nextTwoBodyInfo->tag = j;
        nextTwoBodyInfo->r = rij;
        nextTwoBodyInfo->fcut = fun_cutoff(rij,6.0,6.0);
        nextTwoBodyInfo->fcut_dev = fun_cutoff_dev(rij,6.0,6.0);		
        nextTwoBodyInfo->del[0] = jdelx / rij;
        nextTwoBodyInfo->del[1] = jdely / rij;
        nextTwoBodyInfo->del[2] = jdelz / rij;



        for(int kk = 0; kk < numBonds; kk++) {
          const MEAM2Body& bondk = twoBodyInfo[kk];
          double cos_theta = (nextTwoBodyInfo->del[0]*bondk.del[0] +
                              nextTwoBodyInfo->del[1]*bondk.del[1] +
                              nextTwoBodyInfo->del[2]*bondk.del[2]);
        double cos_theta2=cos_theta*cos_theta;
        double fkk_cut=nextTwoBodyInfo->fcut*bondk.fcut;
	    double rik = bondk.r;
        double rsij2 = sqrt(rij*rik);
        double rvij2 = pow(rij-rik,2);
		
        double gcos1 = cos_theta2*pow(cos_theta2-0.25,2);
        //double gcos2 = pow((cos_theta2-0.25)*(cos_theta2-2.0/3),2);
        double gcos2 = cos_theta*(1.0-cos_theta2)*(1.0-0.5*cos_theta2);
        double gcos3 = pow((cos_theta2-0.72)*(cos_theta2-1.0/3),2);

        double G_cos1 =  cos_theta;
        double G_cos2 = (3*cos_theta2-1);
        double G_cos3 = cos_theta*(5*cos_theta2 -3);	

        //*double Hcos1 =  1.0;		
        double Hcos2 =  cos_theta+1.0;
	    double Hcos3 =  cos_theta2+3*cos_theta +3.0;
 	for(int k=0;k<8;k++)
		 {
            double prefactor=fkk_cut*exp(-3.0*rsij2/eta_list[k])/(1+exp(-1.0*(rsij2))); 
            fvect_unit_Eg(k+56) += prefactor*gcos1;
            fvect_unit_Eg(k+64) += prefactor*gcos2;
            fvect_unit_Eg(k+72) += prefactor*gcos3;

            prefactor=fkk_cut*exp(-1.0*rvij2/eta_list[k])/(1+exp(-1.0*(rij)));
            prefactor = prefactor/(1+exp(-1.0*(rik)));			
	        fvect_unit_Eg(k+16) += prefactor*G_cos1;
            fvect_unit_Eg(k+24) += prefactor*G_cos2;
            fvect_unit_Eg(k+32) += prefactor*G_cos3;
			
			
	        double Sommerfeld_factor1 = 1.0*(rij-2.813)/1.40;
			double Sommerfeld_factor2 = 1.0*(rik-2.813)/1.40;
			double vsij = 1.0*pow(Sommerfeld_factor1,2.0)*pow(1.0-Sommerfeld_factor1,2.0);
			double vsik = 1.0*pow(Sommerfeld_factor2,2.0)*pow(1.0-Sommerfeld_factor2,2.0);
			
			prefactor = fkk_cut*exp(-1.0*vsij*vsik/pow(eta_list[k],2));
			if(k%2==0) fvect_unit_Eg(k+96) += 1.0*pow(electron_temperature_factor,2)*prefactor*Hcos2;
			else       fvect_unit_Eg(k+96) += 1.0*pow(electron_temperature_factor,2)*prefactor*Hcos3;

            /*double rsij2 = (rij-eta_list[k])*(rik-eta_list[k]);
            prefactor=fkk_cut*exp(-1.0*rsij2/eta_list[k]);		
            fvect_unit_Eg(k+88) += prefactor*Hcos1;
            fvect_unit_Eg(k+96) += prefactor*Hcos2;
            fvect_unit_Eg(k+104) += prefactor*Hcos3;*/
          }	
		
        }		  
		  
		  
	     numBonds++;
         nextTwoBodyInfo++;

    }
 }


//printf("nbond =%d\n",numBonds); 

double rho_sum = 0.0;
for(int k=0;k<8;k++)
 {	
     double lastp = rho_values[i][k]+1.0;
     rho_values[i][k] = log(lastp);
     fvect_unit_Eg(k+40)= lastp*(rho_values[i][k]-1.0);
	 
	 /*lastp = rho_values[i][k+16]+1.0;
     rho_values[i][k+16] = log(lastp);
     fvect_unit_Eg(k+104)= pow(temperature_factor,2)*lastp*(rho_values[i][k+16]-1.0);
*/

     lastp = rho_values[i][k+8];
     rho_values[i][k+8] = 0.5/sqrt(lastp);     
     fvect_unit_Eg(k+48)= sqrt(lastp);

     //rho_sum +=fvect_rho(k)*rho_values[i][k+16];
		  }
  
/*fvect_unit_Eg(112)=1.0/(1+exp(rho_scale*(rho_sum-rho_cent)));
double F_rho_dev = fvect_unit_Eg(112)*(fvect_unit_Eg(112)-1.0);

for(int k=0;k<8;k++)
   rho_values[i][k+16]=rho_scale*F_rho_dev*fvect_rho(k);
*/

//debugging
/*
if(i<1)
{
	printf("%lg %lg %lg\n",x[i][0],x[i][1],x[i][2]);	

       printf("nbond =%d\n",numBonds); 	

     for(int t=0;t<80;t++)
       printf("%lg ",fvect_unit_Eg(t));
      printf("\n");
}
*/	

    //comm->forward_comm_pair(this);

  for(int tj = 0; tj<eta_num;tj++)
      PotEg += fvect_dev(tj)*fvect_unit_Eg(tj);

  PotEg +=zero_atom_energy;
	 
	if (eflag) {
       if (eflag_global) eng_vdwl += PotEg;
       if (eflag_atom) eatom[i] += PotEg;
        }
 
 
 //calculating forces for pre-atoms

 
 	 for(int jj = 0; jj < numBonds; jj++) {
      const MEAM2Body bondj = twoBodyInfo[jj];
      double rij = bondj.r;
      int j = bondj.tag;	  
      
      MEAM2Body const* bondk = twoBodyInfo;
      for(int kk = 0; kk < jj; kk++, ++bondk) {
	    double rik = bondk->r;
        int k = bondk->tag;

        double cos_theta = (bondj.del[0]*bondk->del[0] +
                            bondj.del[1]*bondk->del[1] +
                            bondj.del[2]*bondk->del[2]);


        double cos_theta2=cos_theta*cos_theta;
        double fjk_cut=bondj.fcut*bondk->fcut;
	    double frjp_cut=bondj.fcut_dev*bondk->fcut;
	    double frkp_cut=bondj.fcut*bondk->fcut_dev;
        double dcosdrj[3],dcosdrk[3];
        costheta_d(cos_theta,bondj.del,rij,bondk->del,rik,dcosdrj,dcosdrk);
	
        /* for(int jk=0;jk<3;jk++)
             {
                dcosdrj[jk]=(bondk->del[jk]-cos_theta*bondj.del[jk])/rij;
                dcosdrk[jk]=(bondj.del[jk]-cos_theta*bondk->del[jk])/rik;
                 }*/
				 
//8-16 colum
         double g1_cos = cos_theta2*pow(cos_theta2-0.25,2);
         double g1_cos_dev = cos_theta*(cos_theta2-0.25)*(6*cos_theta2-0.5);

//16-24 colum
         double g2_cos = cos_theta*(1.0-cos_theta2)*(1.0-0.5*cos_theta2);
         double g2_cos_dev = cos_theta2*(2.5*cos_theta2-4.5)+1.0;
         
         //double g2_cos = pow((cos_theta2-0.25)*(cos_theta2-2.0/3),2);
         //double g2_cos_dev = 4*cos_theta*(cos_theta2-0.25)*(cos_theta2-2.0/3)*(2*cos_theta2-11.0/12);
//24-36 colum	
          double g3_cos = pow((cos_theta2-0.72)*(cos_theta2-1.0/3),2);
          double g3_cos_dev = 4*cos_theta*(cos_theta2-0.72)*(cos_theta2-1.0/3)*(2*cos_theta2-79.0/75); 

//8-16 colum
         double G1_cos = cos_theta;
         double G1_cos_dev = 1.0;

//16-24 colum
         double G2_cos = 3*cos_theta2-1;
         double G2_cos_dev = 6*cos_theta;
//24-36 colum	
         double G3_cos = cos_theta*(5*cos_theta2 -3);
         double G3_cos_dev = 15.0*cos_theta2 -3 ;
		 
		 /*double Hcos1 =  1.0;
         double Hcos1_dev = 0.0;*/
         double Hcos2 =  cos_theta+1.0;
         double Hcos2_dev = 1.0;
         double Hcos3 =  cos_theta2+3*cos_theta +3;
         double Hcos3_dev = 2*cos_theta + 3;
	
		double fj[3]= {0, 0, 0};
		double fk[3]= {0, 0, 0};		
         double fcos_factor_sum = 0.0;
         double fij_sum = 0.0;
         double fik_sum = 0.0;

        double rsij2 = sqrt(rij*rik);
        double rvij2 = pow(rij-rik,2);
		for(int k=0;k<8;k++)
		 {
			double fjk_eta = fjk_cut/eta_list[k];
			//double fjk_eta2 = fjk_cut/eta2_list[k];
/*********************************************************************************************/
            double fqjk_sgm = 1.0/(1+exp(-1.0*(rsij2)));
            double gausp= exp(-3.0*rsij2/eta_list[k])*fqjk_sgm;
            double prefactor_gij = frjp_cut+0.5*sqrt(rik/rij)*(-3*fjk_eta+fjk_cut*(1-fqjk_sgm));
            double prefactor_gik = frkp_cut+0.5*sqrt(rij/rik)*(-3*fjk_eta+fjk_cut*(1-fqjk_sgm));
			

	     double gcos_sum = 0.0;
	     double gcos_dev_sum = 0.0;			 

//16-24 colum
            double gs1_factor = -1.0*fvect_dev(k+56);			
            gcos_sum += gs1_factor*g1_cos;
	    gcos_dev_sum += gs1_factor*g1_cos_dev;
//24-32 colum	
            double gs2_factor = -1.0*fvect_dev(k+64);		
            gcos_sum += gs2_factor*g2_cos;
	    gcos_dev_sum += gs2_factor*g2_cos_dev;
//32-40 colum	
            double gs3_factor = -1.0*fvect_dev(k+72);		
            gcos_sum += gs3_factor*g3_cos;
	    gcos_dev_sum += gs3_factor*g3_cos_dev;	

            double fr_g_factor = gcos_sum*gausp;
	    double fcos_g_factor = gcos_dev_sum*gausp;			
			
/*********************************************************************************************/	 
             double fj_sgm = 1.0/(1+exp(-1.0*(rij)));

             double fk_sgm = 1.0/(1+exp(-1.0*(rik)));
  	         gausp= exp(-1.0*rvij2/eta_list[k])*fj_sgm*fk_sgm;
             double prefactor_Gij = frjp_cut-2.0*(rij-rik)*fjk_eta+fjk_cut*(1-fj_sgm);
             double prefactor_Gik = frkp_cut-2.0*(rik-rij)*fjk_eta+fjk_cut*(1-fk_sgm);
			 
            double G_cos_sum = 0.0;
	        double G_cos_dev_sum = 0.0;
			
//1-8 colum
            double Gs1_factor = -1.0*fvect_dev(k+16);			
            G_cos_sum += Gs1_factor*G1_cos;
	        G_cos_dev_sum += Gs1_factor*G1_cos_dev;
//8-16 colum	
            double Gs2_factor = -1.0*fvect_dev(k+24);		
            G_cos_sum += Gs2_factor*G2_cos;
	        G_cos_dev_sum += Gs2_factor*G2_cos_dev;

//16-24 colum	
            double Gs3_factor = -1.0*fvect_dev(k+32);		
            G_cos_sum += Gs3_factor*G3_cos;
	        G_cos_dev_sum += Gs3_factor*G3_cos_dev;			
 
            double fr_G_factor = G_cos_sum*gausp;
	        double fcos_G_factor = G_cos_dev_sum*gausp;			          
/*********************************************************************************************/	

            double Sommerfeld_factor1 = 1.0*(rij-2.813)/1.40;
			double Sommerfeld_factor2 = 1.0*(rik-2.813)/1.40;
		    double vsij = 1.0*pow(Sommerfeld_factor1,2)*pow(1.0-Sommerfeld_factor1,2.0);
		    double vsik = 1.0*pow(Sommerfeld_factor2,2)*pow(1.0-Sommerfeld_factor2,2.0);
			gausp = exp(-1.0*vsij*vsik/pow(eta_list[k],2));
	    //double vsij_dev = vsik*(2.0*Sommerfeld_factor1*(1.0-Sommerfeld_factor1)*(2.0*Sommerfeld_factor1-1.0))/1.80;
	    //double vsik_dev = vsij*(2.0*Sommerfeld_factor2*(1.0-Sommerfeld_factor2)*(2.0*Sommerfeld_factor2-1.0))/1.80;
	        double vsij_dev = vsik*2.0*Sommerfeld_factor1*(1.0-Sommerfeld_factor1)*(1.0-2.0*Sommerfeld_factor1)/1.40;
            double vsik_dev = vsij*2.0*Sommerfeld_factor2*(1.0-Sommerfeld_factor2)*(1.0-2.0*Sommerfeld_factor2)/1.40;

			 /*double rsij2 = (rij-eta_list[k])*(rik-eta_list[k]);		
  	         gausp= exp(-1.0*rsij2/eta_list[k]);*/
            double prefactor_Sij = frjp_cut-1.0*vsij_dev*fjk_eta;
            double prefactor_Sik = frkp_cut-1.0*vsik_dev*fjk_eta;
			 
            double Sq_cos_sum = 0.0;
	        double Sq_cos_dev_sum = 0.0;

            double S3_factor = -1.0*pow(electron_temperature_factor,2)*fvect_dev(k+96);
			if(k%2==0){
            Sq_cos_sum += S3_factor*Hcos2;
            Sq_cos_dev_sum += S3_factor*Hcos2_dev;
			}
			else
			{
			Sq_cos_sum += S3_factor*Hcos3;
            Sq_cos_dev_sum += S3_factor*Hcos3_dev;	
			}
 
            double fr_Sq_factor = Sq_cos_sum*gausp;
	        double fcos_Sq_factor = Sq_cos_dev_sum*gausp;			          

/********************************************************************************************/
/*             double rsij2 = (rij-eta_list[k])*(rik-eta_list[k]);		
  	        gausp= exp(-1.0*rsij2/eta_list[k]);
             double prefactor_Sij = frjp_cut-1.0*(rik-eta_list[k])*fjk_eta;
             double prefactor_Sik = frkp_cut-1.0*(rij-eta_list[k])*fjk_eta;
			 
             double Sq_cos_sum = 0.0;
	         double Sq_cos_dev_sum = 0.0;
			
//1-8 colum
            double S1_factor = -1.0*fvect_dev(k+88);			
            Sq_cos_sum += S1_factor*Hcos1;
	    Sq_cos_dev_sum += S1_factor*Hcos1_dev;
//8-16 colum	
            double S2_factor = -1.0*fvect_dev(k+96);		
            Sq_cos_sum += S2_factor*Hcos2;
	    Sq_cos_dev_sum += S2_factor*Hcos2_dev;
//16-24 colum
            double S3_factor = -1.0*fvect_dev(k+104);
            Sq_cos_sum += S3_factor*Hcos3;
            Sq_cos_dev_sum += S3_factor*Hcos3_dev;
 
            double fr_Sq_factor = Sq_cos_sum*gausp;
	    double fcos_Sq_factor = Sq_cos_dev_sum*gausp;*/			          
/*********************************************************************************************/	

        fcos_factor_sum +=(fcos_g_factor+fcos_G_factor+fcos_Sq_factor)*fjk_cut;
        fij_sum +=prefactor_gij*fr_g_factor+prefactor_Gij*fr_G_factor+prefactor_Sij*fr_Sq_factor;
        fik_sum +=prefactor_gik*fr_g_factor+prefactor_Gik*fr_G_factor+prefactor_Sik*fr_Sq_factor;
              }
            fj[0]+=fij_sum*bondj.del[0]+fcos_factor_sum*dcosdrj[0];
			fj[1]+=fij_sum*bondj.del[1]+fcos_factor_sum*dcosdrj[1];
			fj[2]+=fij_sum*bondj.del[2]+fcos_factor_sum*dcosdrj[2];
			
			fk[0]+=fik_sum*bondk->del[0]+fcos_factor_sum*dcosdrk[0];
			fk[1]+=fik_sum*bondk->del[1]+fcos_factor_sum*dcosdrk[1];
			fk[2]+=fik_sum*bondk->del[2]+fcos_factor_sum*dcosdrk[2]; 
			


        forces[i][0] -= fj[0];
        forces[i][1] -= fj[1];
        forces[i][2] -= fj[2];			
        forces[j][0] += fj[0];
        forces[j][1] += fj[1];
        forces[j][2] += fj[2];			
			
			
        forces[i][0] -= fk[0];
        forces[i][1] -= fk[1];
        forces[i][2] -= fk[2];			
        forces[k][0] += fk[0];
        forces[k][1] += fk[1];
        forces[k][2] += fk[2];			

     if(evflag) {
          double delta_ij[3];
          double delta_ik[3];
          delta_ij[0] = bondj.del[0] * rij;
          delta_ij[1] = bondj.del[1] * rij;
          delta_ij[2] = bondj.del[2] * rij;
          delta_ik[0] = bondk->del[0] * rik;
          delta_ik[1] = bondk->del[1] * rik;
          delta_ik[2] = bondk->del[2] * rik;
          ev_tally3(i, j, k, 0.0, 0.0, fj, fk, delta_ij, delta_ik);
           }			
        }
     }	
 }
 // Communicate U'(rho) values

  comm->forward_comm_pair(this);
  
  
  for(int ii = 0; ii < inum_full; ii++) {
    int i = ilist_full[ii];
	
	/// calculate atomic temperature //	
	/*double t = 0.0;
	t += 0.5*(v[i][0]*v[i][0]+v[i][1]*v[i][1]+v[i][2]*v[i][2])*mass[type[i]];
	temperature_factor = 0.02*t*mv2toeV/(3.0*Boltzman_constant);*/
	
    double xtmp = x[i][0];
    double ytmp = x[i][1];
    double ztmp = x[i][2];
    int* jlist = firstneigh_full[i];
    int jnum = numneigh_full[i]; 

		   
    for(int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj];
      j &= NEIGHMASK;

      double jdelx = x[j][0] - xtmp;
      double jdely = x[j][1] - ytmp;
      double jdelz = x[j][2] - ztmp;
      double rij_sq = jdelx*jdelx + jdely*jdely + jdelz*jdelz;
 
      if(rij_sq < cutforcesq) {
        double rij = sqrt(rij_sq);
		double fpair = 0.0; 	
        double fc_rij = fun_cutoff(rij,cutoff);
	    double fcut_coeff = -1.0*sin(rij*Pi_value/cutoff)*Pi_value/(2*rij*cutoff);

        for(int k=0;k<8;k++)
	  {
            double exp1 = exp(-1.0*pow(rij/eta_list[k],2));
	        double fpair1_coeff = -2.0*fc_rij/eta2_list[k];
	        double pair_pot1_deriv = (fpair1_coeff+fcut_coeff)*exp1;			

            fpair += fvect_dev(k)*pair_pot1_deriv;


            double exp2 = exp(-1.0*pow(rij/cutoff,2));
            double gx = pow(mu_list[k]-rij,3);
            double gxp = -3.0*pow(mu_list[k]-rij,2);
            double fpair2_coeff =(gxp-2*gx*rij/cutoff2)*fc_rij/rij;
            double fcut2_coeff = fcut_coeff*gx;
            double pair_pot2_deriv = (fpair2_coeff+fcut2_coeff)*exp2*Hx(mu_list[k]-rij);

            fpair += fvect_dev(k+8)*pair_pot2_deriv;

            double exp5 = exp(-1.0*(rij)/eta_list[k]);
            double fpair5_coeff = exp5*fc_rij/(eta_list[k]*rij*(1+exp5));
            double pair_pot5_deriv = (fpair5_coeff+fcut_coeff)/(1+exp5);
            fpair += fvect_dev(k+80)*pow(atom_temperature_factor,2)*pair_pot5_deriv;

			double fg_Somm_factor = 1.0*(rij-mu_list[k])/1.80;
			double fg_Somm = 1.0*pow(fg_Somm_factor,2)*pow(1.0-fg_Somm_factor,2);
      		double fg_Somm_dev = 2.0*fc_rij*fg_Somm_factor*(1.0-fg_Somm_factor)*(1.0-2.0*fg_Somm_factor)/(1.80*rij);
			double fcut6_coeff = fcut_coeff*fg_Somm;
			double pair_pot6_deriv = (fcut6_coeff+fg_Somm_dev)*Hx(mu_list[k]-rij)*Hx(rij-mu_list[k]+1.80);
			
			fpair += fvect_dev(k+88)*pow(electron_temperature_factor,2)*pair_pot6_deriv;

         /**********************************ManyBody******************************************/			
            /*double fr3 = -1.0/eta_list[k];
            double fpair3_coeff = (fr3*(1+0.5*cos(k*rij))-0.5*k*sin(k*rij))*fc_rij/rij;
            double pair_pot3_deriv = (fpair3_coeff+fcut_coeff*(1+0.5*cos(k*rij)))*exp(-rij/eta_list[k]);*/
            double Urho1_prime_ij= 0.5*fvect_dev(k+40)*(rho_values[i][k]+rho_values[j][k]);			
            fpair += Urho1_prime_ij*pair_pot1_deriv;


            double fpair4_coeff = -2.0*(1.0-mu_list[k]/rij)*fc_rij/Re_sq;
            double pair_pot4_deriv = (fpair4_coeff+fcut_coeff)*exp(-1.0*pow((rij-mu_list[k])/Re,2));
	        double Urho2_prime_ij= 0.5*fvect_dev(k+48)*(rho_values[i][k+8]+rho_values[j][k+8]);

           fpair += Urho2_prime_ij*pair_pot4_deriv; 			


            /*double Urho3_prime_ij= 0.5*fvect_dev(k+104)*(rho_values[i][k+16]+rho_values[j][k+16]); 
           fpair +=  pow(temperature_factor,2)*Urho3_prime_ij*pair_pot6_deriv;*/          
            /*double fr6 = -2.0*rij/10.989225;
            double fpair6_coeff = (fr6*sin((2*k+1)*rij)+(2*k+1)*cos((2*k+1)*rij))*fc_rij/rij;
            double pair_pot6_deriv = (fpair6_coeff+fcut_coeff*sin((2*k+1)*rij))*exp(-1.0*pow(rij/3.315,2));
            double Urho3_prime_ij= 0.5*fvect_dev(112)*(rho_values[i][k+16]+rho_values[j][k+16]);             
            fpair += Urho3_prime_ij*pair_pot6_deriv;*/
                }			 
			 
          forces[i][0] += jdelx*fpair;
          forces[i][1] += jdely*fpair;
          forces[i][2] += jdelz*fpair;

          forces[j][0] -= jdelx*fpair;
          forces[j][1] -= jdely*fpair;
          forces[j][2] -= jdelz*fpair;	
		  
        //if (evflag) ev_tally_full(i,0.0, 0.0, -fpair, jdelx, jdely, jdelz);
		 if (evflag) ev_tally(i,j,nlocal,newton_pair,0.0,0.0,-fpair,jdelx, jdely, jdelz);	 
	      }

	  }
  }


 if(vflag_fdotr) virial_fdotr_compute();
}

/* ---------------------------------------------------------------------- */

void PairMLEnergy::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  map = new int[n+1];

  
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairMLEnergy::settings(int narg, char **arg)
{
  if(narg != 3) error->all(FLERR,"Illegal pair_style command");
  train_flag = atoi(arg[0]);
  zero_atom_energy = atof(arg[1]);
  electron_temperature_factor = atof(arg[2]);
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairMLEnergy::coeff(int narg, char **arg)
{
  int i,j,n;

  if (!allocated) allocate();

  if (narg != 3 + atom->ntypes)
    error->all(FLERR,"Incorrect args for pair coefficients");

  // insure I,J args are * *

  if (strcmp(arg[0],"*") != 0 || strcmp(arg[1],"*") != 0)
    error->all(FLERR,"Incorrect args for pair coefficients");

  // read args that map atom types to elements in potential file
  // map[i] = which element the Ith atom type is, -1 if NULL
  // nelements = # of unique elements
  // elements = list of element names

  if (elements) {
    for (i = 0; i < nelements; i++) delete [] elements[i];
    delete [] elements;
  }
  elements = new char*[atom->ntypes];
  for (i = 0; i < atom->ntypes; i++) elements[i] = NULL;

  nelements = 0;
  for (i = 3; i < narg; i++) {
    if (strcmp(arg[i],"NULL") == 0) {
      map[i-2] = -1;
      continue;
    }
    for (j = 0; j < nelements; j++)
      if (strcmp(arg[i],elements[j]) == 0) break;

    map[i-2] = j;

    if (j == nelements) {
      n = strlen(arg[i]) + 1;
      elements[j] = new char[n];
      strcpy(elements[j],arg[i]);
      nelements++;
    }
  }

/* 
 // for now, only allow single element
  if (nelements > 1)
    error->all(FLERR,
               "Pair meam/spline only supports single element potentials");
     */
	 
	 
  // read potential file
  if(train_flag==0)
   {
     read_file(arg[2]);
     Data_Fitting();
      }
   else 
     read_param(arg[2]);

  // clear setflag since coeff() called once with I,J = * *

  n = atom->ntypes;
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  // set setflag i,j for type pairs where both are mapped to elements

  int count = 0;
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      if (map[i] >= 0 && map[j] >= 0) {
        setflag[i][j] = 1;
        count++;
      }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairMLEnergy::read_file(const char* filename)
{
        if(comm->me == 0) {
                FILE *fp = force->open_potential(filename);
                if(fp == NULL) {
                        char str[1024];
                        sprintf(str,"Cannot open machine learning trainning database file %s", filename);
                        error->one(FLERR,str);
                }

                // Skip first two line of file.
                char line[MAXLINE];

                fgets(line, MAXLINE, fp);
		fgets(line, MAXLINE, fp);
				
		fgets(line, MAXLINE, fp);
        sscanf(line,"%d %d %d",&nfeature,&ntarget,&nlines);
		fclose(fp);
		}	

        MPI_Bcast(&nfeature, 1, MPI_INT, 0, world);
        MPI_Bcast(&ntarget, 1, MPI_INT, 0, world);
        MPI_Bcast(&nlines, 1, MPI_INT, 0, world);		
	    
        // double target_unit;
        Fvect_mpi_arr= (double *)malloc((nlines*nfeature+1)*sizeof(double));
	    Target_mpi_arr= (double *)malloc((nlines*ntarget+1)*sizeof(double));
			 
      if(comm->me == 0) {
         	int iat=0;
		int jat=0;	
		char line[MAXLINE];	
		char *ptr;
		
		FILE *fp = force->open_potential(filename);
        if(fp == NULL) {
            char str[1024];
            sprintf(str,"Cannot open machine learning trainning database file %s", filename);
            error->one(FLERR,str);
                }
		

        fgets(line, MAXLINE, fp);
		fgets(line, MAXLINE, fp);				
		fgets(line, MAXLINE, fp);
		
         for(int i=0;i<nlines;i++)
          {
              fgets(line, MAXLINE, fp);
		 
              ptr = strtok(line," \t\n\r\f");
	
         for(int j=0;j<nfeature;j++)			 
		  {
                    //fvect_unit(j) = atof(ptr);
		    Fvect_mpi_arr[iat++] = atof(ptr);
		    ptr = strtok(NULL," \t\n\r\f");
		  }
		  
		  Target_mpi_arr[jat++] = atof(ptr);
			
            //fvect_arr.push_back(fvect_unit);
            //target_arr.push_back(target_unit);
                } 
        fclose(fp);
        }

        // Transfer training data from master processor to all other processors.
         MPI_Bcast(Fvect_mpi_arr, nlines*nfeature+1, MPI_DOUBLE, 0, world);
	     MPI_Bcast(Target_mpi_arr, nlines*ntarget+1, MPI_DOUBLE, 0, world);

		 
    for(int i=0;i<nlines;i++)
	{
		for(int j=0;j<nfeature;j++)
			fvect_unit(j) = Fvect_mpi_arr[i*nfeature+j];
		fvect_arr.push_back(fvect_unit);
		target_arr.push_back(Target_mpi_arr[i]);
	}		
        // Calculate 'zero-point energy' of single atom in vacuum.
        //zero_atom_energy = 0.0;

        // Determine maximum cutoff radius of all relevant spline functions.
        cutoff = 8.0;
        cutoff2 = cutoff*cutoff;        
 
	eta_num = nfeature;
    memory->create(eta_list,eta_num+1,"pair_ml:eta_list");	
    memory->create(eta2_list,eta_num+1,"pair_ml:eta_list");		
    for(int k=0;k<eta_num;k++)
        {  
	      eta_list[k]=1.0*pow(1.3459,1.0*k);
		  eta2_list[k]= eta_list[k]*eta_list[k];
        } 
	   
        // Set LAMMPS pair interaction flags.
        for(int i = 1; i <= atom->ntypes; i++) {
                for(int j = 1; j <= atom->ntypes; j++) {
                        setflag[i][j] = 1;
                        cutsq[i][j] = cutoff;
                }
        }
}


void PairMLEnergy::read_param(const char* filename)
{
        if(comm->me == 0) {
                FILE *fp = force->open_potential(filename);
                if(fp == NULL) {
                        char str[1024];
                        sprintf(str,"Cannot open machine learning trainning database file %s", filename);
                        error->one(FLERR,str);
                }

                // Skip first two line of file.
                char line[MAXLINE];

        fgets(line, MAXLINE, fp);
		fgets(line, MAXLINE, fp);
				
		fgets(line, MAXLINE, fp);
        sscanf(line,"%d %d %d",&nfeature,&nrho,&ntarget);
		fclose(fp);
		}	

        MPI_Bcast(&nfeature, 1, MPI_INT, 0, world);
        MPI_Bcast(&nrho, 1, MPI_INT, 0, world);
        MPI_Bcast(&ntarget, 1, MPI_INT, 0, world);
  
        printf("Reading Param_files: %d %d %d\n",nfeature,nrho,ntarget); 
        Fvect_mpi_arr= (double *)malloc((nfeature+nrho+3)*sizeof(double));
	    Target_mpi_arr= (double *)malloc((ntarget+1)*sizeof(double));
			 
      if(comm->me == 0) {
        int iat=0;
		int jat=0;	
		char line[MAXLINE];	
		char *ptr;
		
		FILE *fp = force->open_potential(filename);
        if(fp == NULL) {
            char str[1024];
            sprintf(str,"Cannot open machine learning trainning database file %s", filename);
            error->one(FLERR,str);
                }
		

        fgets(line, MAXLINE, fp);
		fgets(line, MAXLINE, fp);				
		fgets(line, MAXLINE, fp);
		
         for(int i=0;i<nfeature+nrho+3;i++)
          {
              fgets(line, MAXLINE, fp);		 
              sscanf(line,"%lf",&Fvect_mpi_arr[i]);	
		  }
		  
              fgets(line, MAXLINE, fp);
              sscanf(line,"%lf",&Target_mpi_arr[0]);
              
        fclose(fp);
        }
   
        // Transfer training data from master processor to all other processors.
         MPI_Bcast(Fvect_mpi_arr, nfeature+nrho+3, MPI_DOUBLE, 0, world);
	     MPI_Bcast(Target_mpi_arr, ntarget+1, MPI_DOUBLE, 0, world);

		 
       for(int j=0;j<nfeature;j++)
	       fvect_dev(j) = Fvect_mpi_arr[j];

       for(int j=0;j<nrho;j++)
               fvect_rho(j) = Fvect_mpi_arr[j+nfeature];

       rho_cent = Fvect_mpi_arr[nfeature+nrho];
       rho_scale = Fvect_mpi_arr[nfeature+nrho+1];
       zero_atom_energy = -1.0*Target_mpi_arr[0];
   

        // Determine maximum cutoff radius of all relevant spline functions.
        cutoff = 8.0;
        cutoff2 = cutoff*cutoff;        
 
	   eta_num = nfeature;
       memory->create(eta_list,eta_num+1,"pair_ml:eta_list");
       memory->create(eta2_list,eta_num+1,"pair_ml:eta_list");	   
       for(int k=0;k<eta_num;k++)
	   {
	        eta_list[k]=1.0*pow(1.3459,1.0*k);
		    eta2_list[k]= eta_list[k]*eta_list[k];
	   }
	   
        // Set LAMMPS pair interaction flags.
        for(int i = 1; i <= atom->ntypes; i++) {
                for(int j = 1; j <= atom->ntypes; j++) {
                        setflag[i][j] = 1;
                        cutsq[i][j] = cutoff;
                }
        }

   //cout<<fvect_dev<<endl;
}

void PairMLEnergy::grab(FILE *fptr, int n, double *list)
{
  char *ptr;
  char line[MAXLINE];

  int i = 0;
  while (i < n) {
    fgets(line,MAXLINE,fptr);
    ptr = strtok(line," \t\n\r\f");
    list[i++] = atof(ptr);
    while (ptr = strtok(NULL," \t\n\r\f")) list[i++] = atof(ptr);
  }
}


void PairMLEnergy::Data_Fitting()
{
   randomize_samples(fvect_arr, target_arr);	
   normalizer.train(fvect_arr);
 
   for (unsigned long i = 0; i < fvect_arr.size(); ++i)
       fvect_arr[i] = normalizer(fvect_arr[i]);
	
  if(comm->me == 0)
        cout << "doing a grid cross-validation" << endl;

    matrix<double> params = logspace(log10(1e-6),log10(1e0),7);
     
    matrix<double> best_result(2,1);
    best_result = 0;
    double best_lambda = 0.000001;
    for(long col =0 ;col <params.nc(); ++col)
    {
        // tell the trainer the parameters we want to use
          const double lambda = params(0,col);


        krr_trainer<kernel_type> trainer_cv;
        trainer_cv.set_lambda(lambda);

        matrix<double> result = cross_validate_regression_trainer(trainer_cv, fvect_arr, target_arr,5);

       if(sum(result)> sum(best_result))
          {
             best_result = result;
             best_lambda = lambda;
              }
    }
     if(comm->me == 0) {
             cout <<"\n best result of grid search: " <<sum(best_result) <<endl;
             cout <<"  best lambda: "<<best_lambda<<endl;
	}
    trainer.set_lambda(best_lambda);
    final_pot_trainer = trainer.train(fvect_arr, target_arr);

   //calculate the derivate of kernel	
   	for(int j=0;j<nfeature;j++)
	   fvect_dev(j) = 0.0;

    for(int i=0;i<final_pot_trainer.basis_vectors.nr();i++)
       fvect_dev += final_pot_trainer.alpha(i)*final_pot_trainer.basis_vectors(i);
   
   double inter_p = final_pot_trainer.b;
   for(int j=0;j<nfeature;j++){
       fvect_dev(j) = fvect_dev(j)*normalizer.std_devs()(j);
     inter_p += normalizer.means()(j)*fvect_dev(j);
   }   

    zero_atom_energy = -1.0*inter_p;   

  if(comm->me == 0) {
    FILE * fp= fopen("Param_ML_pot.txt","w");
    fprintf(fp,"# Fitted ML parameters\n");
    fprintf(fp,"# Zr 91.22\n");
    fprintf(fp,"%d %d\n",nfeature, ntarget);    
    for(int j=0;j<nfeature;j++)
      fprintf(fp,"%lg\n",fvect_dev(j));
    fprintf(fp,"%lg\n",inter_p);
    fclose(fp);
   }  
   //cout<<fvect_dev<<endl;
  // cout<<final_pot_trainer.b<<endl;
}

double PairMLEnergy::fun_cutoff(double r,double Rc)
{
	if(r>Rc)
		return 0.0;
	else
		return 0.5*(cos(Pi_value*r/Rc)+1.0);
}


double PairMLEnergy::fun_cutoff(double r,double Rc,double gramma)
{
	
	if(r>Rc)
		return 0.0;
	else
		return 1+ (gramma*r/Rc-gramma -1)*pow(r/Rc,gramma);
}


double PairMLEnergy::fun_cutoff_dev(double r,double Rc,double gramma)
{
	
	if(r>Rc)
		return 0.0;
	else
		return gramma*(gramma+1)*pow(r/Rc,gramma-1)*(-1.0+r/Rc)/Rc;
}

double PairMLEnergy::Hx(double r)
{
    if(r>0)
       return 1.0;
     else
       return 0.0;
}

void PairMLEnergy::costheta_d(double cos_theta, const double rij_hat[3], double rij,
			     const double rik_hat[3], double rik,
			     double *cos_drj, double *cos_drk)
{
  // first element is devative  wrt Rj, second wrt Rk

  vec3_scaleadd(-cos_theta,rij_hat,rik_hat,cos_drj);
  vec3_scale(1.0/rij,cos_drj,cos_drj);
  vec3_scaleadd(-cos_theta,rik_hat,rij_hat,cos_drk);
  vec3_scale(1.0/rik,cos_drk,cos_drk);

}



double PairMLEnergy::LJ_fpair(double r)
{

   double value;
    value = -250.64*pow(r,-5);
    value += -0.664*exp(8.0/r)*pow(r,-2);
   return 0.5*value/r;

}

double PairMLEnergy::LJ_Epair(double r)
{
  double value;
    value = 62.66*pow(r,-4);
    value += 0.083*exp(8.0/r);
   return 0.5*value;
}



/*
double PairMLEnergy::LJ_fpair(double r)
{

   double value;
    value = -250.64*pow(r,-5);
    //value = -105258.61*exp(-1.0*r/0.2631);
   return value/r;

}

double PairMLEnergy::LJ_Epair(double r)
{
  double value;
    value = 62.66*pow(r,-4);
   // value = 27693.54*exp(-1.0*r/0.2613);
   return value;
}*/
/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */
void PairMLEnergy::init_style()
{
        if(force->newton_pair == 0)
                error->all(FLERR,"Pair style ml/energy requires newton pair on");

        // Need both full and half neighbor list.
        int irequest_full = neighbor->request(this);
        neighbor->requests[irequest_full]->id = 1;
        neighbor->requests[irequest_full]->half = 0;
        neighbor->requests[irequest_full]->full = 1;
        int irequest_half = neighbor->request(this);
        neighbor->requests[irequest_half]->id = 2;
        neighbor->requests[irequest_half]->half = 0;
        neighbor->requests[irequest_half]->half_from_full = 1;
        neighbor->requests[irequest_half]->otherlist = irequest_full;
		
  
	
}

/* ----------------------------------------------------------------------
   neighbor callback to inform pair style of neighbor list to use
   half or full
------------------------------------------------------------------------- */
void PairMLEnergy::init_list(int id, NeighList *ptr)
{
        if(id == 1) listfull = ptr;
        else if(id == 2) listhalf = ptr;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */
double PairMLEnergy::init_one(int i, int j)
{
        return cutoff;
}

/* ---------------------------------------------------------------------- */

int PairMLEnergy::pack_forward_comm(int n, int *list, double *buf, 
                                      int pbc_flag, int *pbc)
{
  int i,j,k,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
	for (k = 0; k < 24; k++) 
	  buf[m++] = rho_values[j][k];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void PairMLEnergy::unpack_forward_comm(int n, int first, double *buf)
{
    int i,k,m,last;

  m = 0;
 last = first + n;
  for (i = first; i < last; i++){
   for (k = 0; k < 24; k++) 
      rho_values[i][k] = buf[m++];
    }

  
}

/* ---------------------------------------------------------------------- */

int PairMLEnergy::pack_reverse_comm(int n, int first, double *buf)
{
  int i,k,m,last;

   m = 0;
  last = first + n;
  for (i = first; i < last; i++)
	 for (k = 0; k < 24; k++) 
	  buf[m++] = rho_values[i][k];
  return m;
}

/* ---------------------------------------------------------------------- */

void PairMLEnergy::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,k,m;

  m = 0;
  for (i = 0; i < n; i++) {
      j = list[i];
   for (k = 0; k < 24; k++) 
      rho_values[j][k] = buf[m++];
    }
}

/* ----------------------------------------------------------------------
   Returns memory usage of local atom-based arrays
------------------------------------------------------------------------- */
double PairMLEnergy::memory_usage()
{
        return nmax *24* sizeof(double);
}

