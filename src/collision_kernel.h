/* 

   Note for the user: 

   This file contains the definition of the collision matrix used to
   solve the drag force implictly. This file is added as an #include
   in both the collisions.c and collisions_template.cu files.

*/

// first index = cols, second index = rows,
// e.g. m[p+o*NFLUIDS] => p col and o row.

idm = lxm*id1 + lym*id2 + lzm*id3;

#if defined(STOKESNUMBER) || defined(EPSTEINDRAG)
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
      if ( p > o )  m[p+o*NFLUIDS] = -dt*omega*alpha[p+o*NFLUIDS]*rho_p/rho_o;
      else          m[p+o*NFLUIDS] = -dt*omega*alpha[p+o*NFLUIDS];
#endif

#ifdef CONSTANTDRAG
      m[p+o*NFLUIDS] = -dt*alpha[p+o*NFLUIDS]/rho_o;
#endif

#if defined(EPSTEINDRAG)
      
      rho_gas = 0.5*(rho[0][l] + rho[0][idm]);

#ifdef Z   //3D
      /* in 3D, alpha already constains 1/St/Sigma_gas/(cs/omega)
	 (=sqrt(8/pi)/s/rho_int) via condinit.c. We thus need to
	 multiply alpha by rho_gas c_s / omega to recover the inverse
	 Stokes number. */
#ifdef ISOTHERMAL
      mycs = 0.5*(cs[l] + cs[idm]);
#else
      mycs = 0.5*( sqrt(GAMMA*(GAMMA-1.0)*cs[l]/rho[0][l]) + sqrt(GAMMA*(GAMMA-1.0)*cs[idm]/rho[0][idm]) );
#endif
      myfactor = rho_gas*mycs/omega;
#else      //2D
      /* in 2D, alpha already constains 1/St/Sigma_gas (=
	 2/pi/s/rho_int) via condinit.c. We thus need to multiply
	 alpha by Sigma_gas to recover the inverse Stokes number. */
      myfactor = rho_gas;
#endif
      if ( p > o )  m[p+o*NFLUIDS] = -dt*omega*myfactor*alpha[p+o*NFLUIDS]*rho_p/rho_o;
      else          m[p+o*NFLUIDS] = -dt*omega*myfactor*alpha[p+o*NFLUIDS];
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
	  
#if (defined(STOKESNUMBER) || defined(CONSTANTSTOKESNUMBER)) && !defined(EPSTEINDRAG)
	  /* In the line below, the collision term should be
	     alpha[p+q*NFLUIDS], however, we use alpha[q+p*NFLUIDS] to
	     have the possibility of disabling feedback if necessary.*/
	  
	  if( q > p ) sum += alpha[q+p*NFLUIDS]*rho_q/rho_p;
	  else        sum += alpha[q+p*NFLUIDS];
#endif

#ifdef CONSTANTDRAG
	  sum += alpha[q+p*NFLUIDS];
#endif	  

#if defined(EPSTEINDRAG)
	  
	  rho_gas = 0.5*(rho[0][l] + rho[0][idm]);
	  
#ifdef Z   //3D
	  /* in 3D, alpha already constains 1/St/Sigma_gas/(cs/omega)
	     (=sqrt(8/pi)/s/rho_int) via condinit.c. We thus need to
	     multiply alpha by rho_gas c_s / omega to recover the inverse
	     Stokes number. */
#ifdef ISOTHERMAL
	  mycs = 0.5*(cs[l] + cs[idm]);
#else
	  mycs = 0.5*( sqrt(GAMMA*(GAMMA-1.0)*cs[l]/rho[0][l]) + sqrt(GAMMA*(GAMMA-1.0)*cs[idm]/rho[0][idm]) );
#endif
	  myfactor = rho_gas*mycs/omega;
#else      //2D
	  /* in 2D, alpha already constains 1/St/Sigma_gas (=
	     2/pi/s/rho_int) via condinit.c. We thus need to multiply
	     alpha by Sigma_gas to recover the inverse Stokes number. */
	  myfactor = rho_gas;
#endif
	  if( q > p ) sum += myfactor*alpha[q+p*NFLUIDS]*rho_q/rho_p;
	  else        sum += myfactor*alpha[q+p*NFLUIDS];
#endif
	}
      }
      
#if defined(STOKESNUMBER) || defined(CONSTANTSTOKESNUMBER) || defined(EPSTEINDRAG)
      m[p+p*NFLUIDS] = 1.0 + dt*omega*sum; //The factors were not present in the sum (see **)
#endif
      
#ifdef CONSTANTDRAG
      m[p+p*NFLUIDS] = 1.0 + dt*sum/rho_p; //The factors were not present in the sum (see **)
#endif
    }
  }
  b[o] = velocities_input[o][l];
 }
