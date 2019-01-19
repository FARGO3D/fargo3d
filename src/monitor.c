#include "fargo3d.h"
#define FORMAT "%.12g\t"
#define MAX_MONITOR 20
#define MAX_STRING  100


void (*mon_func[MAX_MONITOR])();
int mon_kind[MAX_MONITOR];
int mon_pldp[MAX_MONITOR];
char mon_name[MAX_MONITOR][MAX_STRING];
char mon_cent[MAX_MONITOR][MAX_STRING];

static boolean func_declared = NO;
static int MonCounter=0;


int Index(int v) {
  unsigned int r = 0;
  while (v >>= 1) {
    r++;
  }
  return r;
}

void Write1DFile (char *filename, real *x, real *y, int n) {
  FILE *out;
  int i;
  out = fopen_prs (filename, "w");
  if (CPU_Rank == 0) {
    for (i = 0; i < n; i++) {
      fprintf (out, "%.15g\t%.15g\n", x[i], y[i]);
    }
  }
  fclose (out);
}
/* Below: 3D monitoring not implemented yet */

void InitMonitoring3D (int bitchoice) {
  int idx, r=1;
  while (bitchoice) {
    if (bitchoice & 1) {
      idx = Index(r);
      //      SetUpMonitoring (idx);
    }
    bitchoice >>= 1;
    r <<= 1;
  }
}

void InitMonitoring() {
  InitFunctionMonitoring (MASS,    mon_dens, "mass", TOTAL, "YCZC", INDEP_PLANET);
  InitFunctionMonitoring (MOM_X,   mon_momx, "momx", TOTAL, "YCZC", INDEP_PLANET);
  InitFunctionMonitoring (MOM_Y,   mon_momy, "momy", TOTAL, "YCZC", INDEP_PLANET);
  InitFunctionMonitoring (MOM_Z,   mon_momz, "momz", TOTAL, "YCZC", INDEP_PLANET);
  InitFunctionMonitoring (TORQ,    mon_torq, "torq", TOTAL, "YCZC", DEP_PLANET);
  InitFunctionMonitoring (REYNOLDS, mon_reynolds, "reynolds", TOTAL, "YCZC", INDEP_PLANET);
  InitFunctionMonitoring (MAXWELL, mon_maxwell, "maxwell", TOTAL, "YCZC", INDEP_PLANET);
  InitFunctionMonitoring (BXFLUX, mon_bxflux, "bxflux", TOTAL, "YCZC", INDEP_PLANET);
  func_declared = YES;
}

void InitFunctionMonitoring (int bittype, void (*f)(), char *name,\
			     int kind, char *centering, int planetdep) {
  int idx;
  idx = Index(bittype);
  if (idx >= MAX_MONITOR) {
    prs_error ("Too many quantities monitored. Rebuild code after increasing MAX_MONITOR in monitor.c\n");
  }
  mon_func[idx] = f;
  mon_kind[idx] = kind;
  mon_pldp[idx] = planetdep;
  strncpy(mon_name[idx], name, MAX_STRING-1);
  strncpy(mon_cent[idx], centering, 5);
}

void MonitorFunction (int idx, int r, char *CurrentFineGrainDir, int plnb) {
  static real Profile[MAX1D];
  static real GProfile[MAX1D];
  static real Coord[MAX1D];
  static real GCoord[MAX1D];
  char filename[MAXLINELENGTH];
  char planet_number[MAXLINELENGTH];
  boolean centered;
  real lsum=0.0, gsum=0.0;
  int j, k;
  FILE *Out;
  if (plnb < 0)
    sprintf (planet_number, "%s", "");
  else
    sprintf (planet_number, "_planet_%d", plnb);
  if (r & MONITOR2D) {
    sprintf (filename, "%s_2d_%07d%s.dat", mon_name[idx], MonCounter, planet_number);
    Write2D (Reduction2D, filename, CurrentFineGrainDir, NOGHOSTINC);
  }
  if ((r & (MONITORY)) | (r & (MONITORY_RAW))) {
    centered = NO;
    if ((mon_cent[idx][1] == 'C') || (mon_cent[idx][1] == 'c'))
      centered = YES;
    INPUT2D (Reduction2D);
    for (j = 0; j < NY; j++) {
      Profile[j] = 0.0;
      Coord[j] = -1e30;
    }
    for (j = NGHY; j < Ny+NGHY; j++) {
      Coord[j+y0cell-NGHY] = (centered ? Ymed(j) : Ymin(j));
      for (k = NGHZ; k < Nz+NGHZ; k++) {
	Profile[j+y0cell-NGHY] += Reduction2D->field_cpu[l2D];
      }
    }
#ifndef FLOAT
    MPI_Reduce (Profile, GProfile, NY, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce (Coord,   GCoord,   NY, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
#else
    MPI_Reduce (Profile, GProfile, NY, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce (Coord,   GCoord,   NY, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
#endif
    // Now GProfile contains the intended 1D profile
    if (r & MONITORY) {
      sprintf (filename, "%s/%s_1d_Y_%07d%s.dat",CurrentFineGrainDir,	\
	       mon_name[idx], MonCounter, planet_number);
      Write1DFile (filename, GCoord, GProfile, NY);
    }
    if (r & MONITORY_RAW) {
      sprintf (filename, "%s/%s_1d_Y_raw%s.dat", OUTPUTDIR, mon_name[idx], planet_number);
      Out = fopen_prs (filename, "a");
      if (CPU_Rank == 0) {
	fwrite (GProfile, sizeof (real), NY, Out);
      }
      fclose (Out);
    }
  }
  if ((r & MONITORZ) | (r & MONITORZ_RAW)) {
    centered = NO;
    if ((mon_cent[idx][3] == 'C') || (mon_cent[idx][3] == 'c'))
      centered = YES;
    INPUT2D (Reduction2D);
    for (j = 0; j < NY; j++) {
      Profile[j] = 0.0;
      Coord[j] = 0.0;
    }
    for (k = NGHZ; k < Nz+NGHZ; k++) {
      Coord[k+z0cell-NGHZ] = (centered ? Zmed(j) : Zmin(j));
      for (j = NGHY; j < Ny+NGHY; j++) {
	Profile[k+z0cell-NGHZ] += Reduction2D->field_cpu[l2D];
      }
    }
#ifndef FLOAT 
    MPI_Reduce (Profile, GProfile, NZ, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce (Coord,   GCoord,   NZ, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
#else
    MPI_Reduce (Profile, GProfile, NZ, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce (Coord,   GCoord,   NZ, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
#endif
    // Now GProfile contains the intended 1D profile
    if (r & MONITORZ) {
      sprintf (filename, "%s/%s_1d_Z_%07d%s.dat",CurrentFineGrainDir,	\
	       mon_name[idx], MonCounter, planet_number);
      Write1DFile (filename, GCoord, GProfile, NZ);
    }
    if (r & MONITORZ_RAW) {
      sprintf (filename, "%s/%s_1d_Z_raw%s.dat", OUTPUTDIR, mon_name[idx], planet_number);
      Out = fopen_prs (filename, "a");
      if (CPU_Rank == 0) {
	fwrite (GProfile, sizeof (real), NZ, Out);
      }
      fclose (Out);
    }
  }
  if (r & MONITORSCALAR) {
    INPUT2D (Reduction2D);
    for (k = NGHZ; k < Nz+NGHZ; k++) {
      for (j = NGHY; j < Ny+NGHY; j++) {
	lsum += Reduction2D->field_cpu[l2D];
      }
    }
#ifndef FLOAT
    MPI_Reduce(&lsum, &gsum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#else
    MPI_Reduce(&lsum, &gsum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
#endif
    sprintf (filename, "%s/%s%s.dat", OUTPUTDIR, mon_name[idx], planet_number);
    Out = fopen_prs (filename, "a");
    if (CPU_Rank == 0) {
      fprintf (Out, "%.12g\t%.12g\n", PhysicalTime, gsum);
    }
    fclose (Out);
  }
}

void DoMonitoring (int idx, int r, char *dir) {
  int nb, i, j;
  nb = Sys->nb;
  if (mon_pldp[idx] == INDEP_PLANET) nb = 1;
  for (i = 0; i < nb; i++) {
    if (ThereArePlanets) {
      Xplanet = Sys->x[i];
      Yplanet  = Sys->y[i];
      Zplanet  = Sys->z[i];
      VXplanet = Sys->vx[i];
      VYplanet = Sys->vy[i];
      VZplanet = Sys->vz[i];
      MplanetVirtual = Sys->mass[i];
    } else {
      Xplanet = Yplanet = Zplanet = VXplanet = VYplanet = VZplanet = MplanetVirtual = 0.0;
    }
    mon_func[idx]();
    reduction_SUM (Slope, NGHY, Ny+NGHY, NGHZ, Nz+NGHZ);
    j = i;
    if (mon_pldp[idx] == INDEP_PLANET) j = -1;
    MonitorFunction (idx, r, dir, j);
  }
}

void MonitorGlobal (int bitchoice) {
  char CurrentFineGrainDir[MAXLINELENGTH];
  int r=1, idx;
  if (func_declared == NO) InitMonitoring ();
  sprintf (CurrentFineGrainDir, "%s/FG%06d/", OUTPUTDIR, TimeStep);
  if (bitchoice & REYNOLDS)
    ComputeVmed (Vx);
  while (bitchoice) {
    if (bitchoice & 1) {
      idx = Index(r);
      DoMonitoring (idx, r, CurrentFineGrainDir);
    }
    bitchoice >>= 1;
    r <<= 1;
  }
  MonCounter++;
}
