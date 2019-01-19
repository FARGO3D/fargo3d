#include "fargo3d.h"

void buildprime(int *prime) {
  boolean isprime[MAXPRIME];
  int i, j;
  for (i = 0; i < MAXPRIME; i++) {
    isprime[i] = TRUE;
    prime[i] = 0;
  }
  for (i = 2; i < MAXPRIME; i++) {
    if (isprime[i]) {
      for (j = 2*i; j < MAXPRIME; j++) {
	if (!(j%i)) isprime[j] = FALSE;
      }
    }
  }
  isprime[0] = isprime[1] = j = 0;
  for (i = 2; i < MAXPRIME; i++) {
    if (isprime[i]) prime[j++] = i;
  }
}

void primefactors (int n, int *factors, int *nfact) {
  int j=0, k=0;
  int prime[MAXPRIME];
  buildprime(prime);
  while (n > 1) {
    if (n % prime[j]) { // Remainder of n/prime[j] True if n%prime[j]>0
      j++;
    }
    else {
      n /= prime[j];
      factors[k++] = prime[j];
    }
  }
  *nfact = k;
}

void repartition (int *nx, int ncpu, int *MX) {
  int factors[MAXPRIME];
  int nfact;
  int i, j;
  int ncomb;
  int pow2[MAXPRIME], comb[MAXPRIME];
  int nb;
  int mx[2], idx;
  real best = 1e30;

  primefactors (ncpu, factors, &nfact); // Decompose ncpu into prime factors
  for (i = 0; i <= nfact; i++) {
    pow2[i] = (int)(pow(2.0,(real)i)+.01); //Storing powers of 2
  }
  ncomb = pow2[nfact]; // Number of possibilities 2^(#fact)
  for (i=0; i < ncomb; i++) {
    nb = i; 
    for (j = nfact-1; j >= 0; j--) {
      comb[j] = nb/pow2[j];  
      nb -= comb[j]*pow2[j];
    }
    for (j = 0; j<2; j++) mx[j] = 1;
    for (j = 0; j < nfact; j++)
      mx[comb[j]] *= factors[j];
    idx = mx[0]*nx[1]+mx[1]*nx[0];
    if (idx < best) {
      best = idx;
      for (j = 0; j < 2; j++) MX[j] = mx[j];
    }
  }
#ifdef DEBUG
  if(!CPU_Rank)  printf("CPU_GRID = %d*%d\n", MX[0], MX[1]);
#endif
  Ncpu_x = MX[0];
  Ncpu_y = MX[1];
}

void split(Grid *g) {
  int  ix, iy;
  int tamanyox, tamanyoy;
  int ixceldas, iyceldas;
  int resto, cociente;
  int nx[2];
  int MX[2];

#ifdef DEBUG
  char filename[200];
  FILE *grid_file;
  sprintf(filename, "%sgrid%03d.inf", OUTPUTDIR, CPU_Rank);
  grid_file = fopen_prs (filename, "w");
#endif

  nx[0] = NY;
  nx[1] = NZ;

  repartition (nx, CPU_Number, MX);

  ix = CPU_Rank % MX[0]; //Coordinates of the grid of CPUs
  iy = CPU_Rank / MX[0];

// |-----+-----+-----+-----|
// | ... | ... | ... | ... |
// |-----+-----+-----+-----|
// | 1,0 | 1,1 | 1,2 | 1,3 |
// |-----+-----+-----+-----|
// | 0,0 | 0,1 | 0,2 | 0,3 |
// |-----+-----+-----+-----|

  ix = CPU_Rank % MX[0];
  iy = CPU_Rank / MX[0];
  resto = nx[0] % MX[0];
  cociente = nx[0] / MX[0];
  //Round robin size distribution
  if (ix < resto) {
    tamanyox = cociente + 1;
    ixceldas = ix*(cociente+1);
  } else {
    tamanyox = cociente;
    ixceldas = ix*cociente+resto;
  }

  resto = nx[1] % MX[1];
  cociente = nx[1] / MX[1];

  if (iy < resto) {
    tamanyoy = cociente + 1;
    iyceldas = iy*(cociente+1);
  } else {
    tamanyoy = cociente;
    iyceldas = iy*cociente+resto;
  }

  Gridd.nx = NX;
  Gridd.ny = tamanyox;
  Gridd.nz = tamanyoy;
  Gridd.J = ix;
  Gridd.K = iy;
  Gridd.NJ = MX[0];
  Gridd.NK = MX[1];
  Gridd.stride = (Gridd.nx+2*NGHX)*(Gridd.ny+2*NGHY);
  Nx = Gridd.nx;
  Ny = Gridd.ny;
  Nz = Gridd.nz;
  J  = Gridd.J;
  K  = Gridd.K;
  Stride = Gridd.stride;
  y0cell = ixceldas;
  z0cell = iyceldas;
  Y0 = y0cell;
  Z0 = z0cell;

#ifdef DEBUG
  fprintf(grid_file, "CPU_Rank\tY0\tYN\tZ0\tZN\tIndexY\tIndexZ\n");
  fprintf(grid_file, "%d\t%d\t%d\t%d\t%d\t%d\t%d\n",CPU_Rank, \
  	  y0cell, y0cell+tamanyox-1, z0cell ,z0cell+tamanyoy-1, J, K);
  fclose(grid_file);
#endif
  
  Xmin = (real *)malloc(sizeof(real)*(Nx+2*NGHX+1));
  Ymin = (real *)malloc(sizeof(real)*(Ny+2*NGHY+1));
  Zmin = (real *)malloc(sizeof(real)*(Nz+2*NGHZ+1));

  Xmed = (real *)malloc(sizeof(real)*(Nx+2*NGHX));
  Ymed = (real *)malloc(sizeof(real)*(Ny+2*NGHY));
  Zmed = (real *)malloc(sizeof(real)*(Nz+2*NGHZ));

  InvDiffXmed = (real *)malloc(sizeof(real)*(Nx+2*NGHX));
  InvDiffYmed = (real *)malloc(sizeof(real)*(Ny+2*NGHY));
  InvDiffZmed = (real *)malloc(sizeof(real)*(Nz+2*NGHZ));

  Sxj  = (real *)malloc(sizeof(real)*(Ny+2*NGHY));
  Syj  = (real *)malloc(sizeof(real)*(Ny+2*NGHY));
  Szj  = (real *)malloc(sizeof(real)*(Ny+2*NGHY));

  Sxk  = (real *)malloc(sizeof(real)*(Nz+2*NGHZ));
  Syk  = (real *)malloc(sizeof(real)*(Nz+2*NGHZ));
  Szk  = (real *)malloc(sizeof(real)*(Nz+2*NGHZ));

  InvVj = (real *)malloc(sizeof(real)*(Ny+2*NGHY));
  
  ycells = ixceldas;
  zcells = iyceldas;

  Gridd.bc_up    = ((K == Ncpu_y-1) && (!PERIODICZ) ? 1 : 0);
  Gridd.bc_down  = ((K == 0)        && (!PERIODICZ) ? 1 : 0);
  Gridd.bc_right = ((J == Ncpu_x-1)  ? 1 : 0);
  Gridd.bc_left  = ((J == 0)         ? 1 : 0);

#ifdef GPU

  DevMalloc(&Xmin_d,sizeof(real)*(Nx+2*NGHX+1));
  DevMalloc(&Ymin_d,sizeof(real)*(Ny+2*NGHY+1));
  DevMalloc(&Zmin_d,sizeof(real)*(Nz+2*NGHZ+1));

  DevMalloc(&Sxj_d,sizeof(real)*(Ny+2*NGHY));
  DevMalloc(&Syj_d,sizeof(real)*(Ny+2*NGHY));
  DevMalloc(&Szj_d,sizeof(real)*(Ny+2*NGHY));

  DevMalloc(&Sxk_d,sizeof(real)*(Nz+2*NGHZ));
  DevMalloc(&Syk_d,sizeof(real)*(Nz+2*NGHZ));
  DevMalloc(&Szk_d,sizeof(real)*(Nz+2*NGHZ));

  DevMalloc(&InvVj_d,sizeof(real)*(Ny+2*NGHY));

  DevMalloc(&Alpha_d,sizeof(real)*NFLUIDS*NFLUIDS);

#endif
  
  //We allocate and initialize the memory of the collision matrix
  Alpha = (real*) calloc(NFLUIDS*NFLUIDS,sizeof(real));

}
