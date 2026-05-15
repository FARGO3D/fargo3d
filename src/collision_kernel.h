/* 

   Note for the user: 

   This file contains the definition of the collision matrix used to
   solve the drag force implictly. This file is added as an #include
   in both the collisions.c and collisions_template.cu files.

*/

// first index = cols, second index = rows,
// e.g. m[p+o*NFLUIDS] => p col and o row.

idm = lxm*id1 + lym*id2 + lzm*id3;

#ifdef STOKESNUMBER
omega = (id1+id3)*sqrt(G*MSTAR/(ymed(j)*ymed(j)*ymed(j))) +
  id2*sqrt(G*MSTAR/(ymin(j)*ymin(j)*ymin(j)));
#endif

#ifdef CONSTANTSTOKESNUMBER
#ifdef SHEARINGBOX
omega = OMEGAFRAME;
#else
omega = 1.0;
#endif
#endif


// In the implementation below, alpha --> 1/St

for (o=0; o<NFLUIDS; o++) {
  for (p=0; p<NFLUIDS; p++) {
    
    rho_p  = 0.5*(rho[p][l] + rho[p][idm]);
    
    // off-diagonal elements
    if (p != o) {
      
      rho_o  = 0.5*(rho[o][l] + rho[o][idm]);
      
      /* In the line below, the collision term should be
	 alpha[o+p*NFLUIDS], however, we use alpha[p+o*NFLUIDS] to
	 have the possibility of disabling feedback if necessary.*/      
      
#if defined(STOKESNUMBER) || defined(CONSTANTSTOKESNUMBER)
//      if ( p > o )  m[p+o*NFLUIDS] = -dt*omega*alpha[p+o*NFLUIDS]*rho_p/rho_o;
//     else          m[p+o*NFLUIDS] = -dt*omega*alpha[p+o*NFLUIDS];
      if ( p > o )  m[p+o*NFLUIDS] = 0.0;
      else {
	      if (p == 0) {
		      if (o == 1) m[p+o*NFLUIDS] = -dt*omega*alpha[p+o*NFLUIDS];
		      else m[p+o*NFLUIDS] = -dt*omega*2.0*(0.5*(rho[0][l] + rho[0][idm]))/M_PI/(RHOS/MUNIT*LUNIT*LUNIT*LUNIT)/((alpha[p+o*NFLUIDS]+1.0e-30)/LUNIT);
              }
	      else m[p+o*NFLUIDS] = 0.0;
      }	     

#endif
#ifdef CONSTANTDRAG
      m[p+o*NFLUIDS] = -dt*alpha[p+o*NFLUIDS]/rho_o;
#endif
    }
    
    // diagonal elements
    else {
      
      /* We now compute the sum present in the diagonal elements.
	 (**) The sum is factorized by dt*omega  */
      
      sum = 0.0;
      for (q=0; q<NFLUIDS; q++) {
	
	//Element pp not included
	if (q != p){
	  
	  rho_q  = 0.5*(rho[q][l] + rho[q][idm]);
	  
#if defined(STOKESNUMBER) || defined(CONSTANTSTOKESNUMBER)
	  
	  /* In the line below, the collision term should be
	     alpha[p+q*NFLUIDS], however, we use alpha[q+p*NFLUIDS] to
	     have the possibility of disabling feedback if necessary.*/
	  
//	  if( q > p ) sum += alpha[q+p*NFLUIDS]*rho_q/rho_p;
//	  else        sum += alpha[q+p*NFLUIDS];
          if ( q > p )  sum += 0.0;
          else {
                  if (q == 0) {
			  if (p == 1) sum += alpha[q+p*NFLUIDS];
                          else sum += 2.0*(0.5*(rho[0][l] + rho[0][idm]))/M_PI/(RHOS/MUNIT*LUNIT*LUNIT*LUNIT)/((alpha[q+p*NFLUIDS]+1.0e-30)/LUNIT);
		  }
                  else sum += 0.0;
          }

#endif
#ifdef CONSTANTDRAG
	  sum += alpha[q+p*NFLUIDS];
#endif	  
	}
      }
      
#if defined(STOKESNUMBER) || defined(CONSTANTSTOKESNUMBER)
      m[p+p*NFLUIDS] = 1.0 + dt*omega*sum; //The factors were not present in the sum (see **)
#endif
      
#ifdef CONSTANTDRAG
      m[p+p*NFLUIDS] = 1.0 + dt*sum/rho_p; //The factors were not present in the sum (see **)
#endif
    }
  }
  b[o] = velocities_input[o][l];
 }
