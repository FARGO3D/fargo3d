#include "fargo3d.h"

//#ifndef MPI
//#define MPI_COMM_WORLD 0
//#define MPI_INT 0
//#define MPI_CHAR 1
//#endif


// 匹配 10 列数据格式
typedef struct {
    int index;
    real x, y, z, vx, vy, vz, mp, time, omega;
} PlanetSnapshot;

// 使用指针数组支持多个行星，MAX_PLANETS 取 10 或根据需求定义
#define MAX_TRACKED_PLANETS 10
static PlanetSnapshot *planets_data[MAX_TRACKED_PLANETS] = {NULL};
static int lines_per_planet[MAX_TRACKED_PLANETS] = {0};
static int last_indices[MAX_TRACKED_PLANETS] = {0};

double HermiteInterpolate(double p0, double p1, double v0, double v1, double dt, double t_frac) {
    double t2 = t_frac * t_frac;
    double t3 = t2 * t_frac;
    
    // Hermite 基函数
    double h00 = 2*t3 - 3*t2 + 1;
    double h10 = t3 - 2*t2 + t_frac;
    double h01 = -2*t3 + 3*t2;
    double h11 = t3 - t2;
    
    // 注意速度项需要乘上 dt，因为导数是相对于物理时间的
    return h00 * p0 + h10 * dt * v0 + h01 * p1 + h11 * dt * v1;
}

void UpdatePlanetFromTrajectory(PlanetarySystem *sys, int k, real current_time) {
    // --- 1. 初始化与并行读取 (每个行星只在第一次调用时读取一次) ---
	int i;
    if (planets_data[k] == NULL) {
        if (CPU_Rank == 0) {
            char filename[1024];
            // 动态生成文件名，如 ibigplanet0.dat, ibigplanet1.dat
            sprintf(filename, "%sibigplanet%d.dat", OUTPUTDIR, k);
            
            FILE *fp = fopen(filename, "r");
            if (!fp) {
                mastererr("Error: Cannot open %s\n", filename);
                prs_exit(1);
            }
            
            int n_lines = 0;
            char line[1024];
            while (fgets(line, sizeof(line), fp)) n_lines++;
            rewind(fp);

            planets_data[k] = (PlanetSnapshot *)malloc(n_lines * sizeof(PlanetSnapshot));
            for (i = 0; i < n_lines; i++) {
                fscanf(fp, "%d %lf %lf %lf %lf %lf %lf %lf %lf %lf", 
                       &planets_data[k][i].index, &planets_data[k][i].x, &planets_data[k][i].y, 
                       &planets_data[k][i].z, &planets_data[k][i].vx, &planets_data[k][i].vy, 
                       &planets_data[k][i].vz, &planets_data[k][i].mp, &planets_data[k][i].time, 
                       &planets_data[k][i].omega);
            }
            fclose(fp);
            lines_per_planet[k] = n_lines;
        }

        // MPI 广播该行星的数据行数
        MPI_Bcast(&lines_per_planet[k], 1, MPI_INT, 0, MPI_COMM_WORLD);

        // 非主进程分配内存
        if (CPU_Rank != 0) {
            planets_data[k] = (PlanetSnapshot *)malloc(lines_per_planet[k] * sizeof(PlanetSnapshot));
        }

        // 广播该行星的完整轨迹数组
        MPI_Bcast(planets_data[k], lines_per_planet[k] * sizeof(PlanetSnapshot), MPI_CHAR, 0, MPI_COMM_WORLD);
    }

    // --- 2. 变量定义与边界处理 ---
    int n = lines_per_planet[k];
    int idx = last_indices[k];
    real t0, t1, f;

    // A. 时间未到：停在第一行
    if (current_time <= planets_data[k][0].time) {
        f = 0.0;
        t0 = planets_data[k][0].time;
        t1 = planets_data[k][0].time;
        idx = 0;
    } 
    // B. 时间超过：停在最后一行
    else if (current_time >= planets_data[k][n-1].time) {
        f = 0.0;
        t0 = planets_data[k][n-1].time;
        t1 = planets_data[k][n-1].time;
        idx = n - 1;
    }
    // C. 正常插值区间定位
    else {
        while (idx < n - 2 && planets_data[k][idx + 1].time < current_time) {
            idx++;
        }
        last_indices[k] = idx; // 更新缓存索引
        t0 = planets_data[k][idx].time;
        t1 = planets_data[k][idx + 1].time;
        
        // 防除以 0 检查
        if (t1 == t0) f = 0.0;
        else f = (current_time - t0) / (t1 - t0);
    }

	real dt = t1 - t0;

    // --- 3. 应用插值结果到系统 ---
	sys->x[k] = HermiteInterpolate(planets_data[k][idx].x,  planets_data[k][idx+1].x, 
                               planets_data[k][idx].vx, planets_data[k][idx+1].vx, dt, f);

	sys->y[k] = HermiteInterpolate(planets_data[k][idx].y,  planets_data[k][idx+1].y, 
                               planets_data[k][idx].vy, planets_data[k][idx+1].vy, dt, f);

	sys->z[k] = HermiteInterpolate(planets_data[k][idx].z,  planets_data[k][idx+1].z, 
                               planets_data[k][idx].vz, planets_data[k][idx+1].vz, dt, f);
	sys->vx[k] = planets_data[k][idx].vx + f * (planets_data[k][idx+1].vx - planets_data[k][idx].vx);
	sys->vy[k] = planets_data[k][idx].vy + f * (planets_data[k][idx+1].vy - planets_data[k][idx].vy);
	sys->vz[k] = planets_data[k][idx].vz + f * (planets_data[k][idx+1].vz - planets_data[k][idx].vz);
	
    //sys->x[k]  = planets_data[k][idx].x  + f * (planets_data[k][idx+1].x  - planets_data[k][idx].x);
    //sys->y[k]  = planets_data[k][idx].y  + f * (planets_data[k][idx+1].y  - planets_data[k][idx].y);
    //sys->z[k]  = planets_data[k][idx].z  + f * (planets_data[k][idx+1].z  - planets_data[k][idx].z);
    //sys->vx[k] = planets_data[k][idx].vx + f * (planets_data[k][idx+1].vx - planets_data[k][idx].vx);
    //sys->vy[k] = planets_data[k][idx].vy + f * (planets_data[k][idx+1].vy - planets_data[k][idx].vy);
    //sys->vz[k] = planets_data[k][idx].vz + f * (planets_data[k][idx+1].vz - planets_data[k][idx].vz);
    
}

void ComputeIndirectTerm () {
#ifndef NODEFAULTSTAR
  IndirectTerm.x = -DiskOnPrimaryAcceleration.x;
  IndirectTerm.y = -DiskOnPrimaryAcceleration.y;
  IndirectTerm.z = -DiskOnPrimaryAcceleration.z;
  if (!INDIRECTTERM) {
    IndirectTerm.x = 0.0;
    IndirectTerm.y = 0.0;
    IndirectTerm.z = 0.0;
  }
#else
  IndirectTerm.x = 0.0;
  IndirectTerm.y = 0.0;
  IndirectTerm.z = 0.0;
#endif
}

Force ComputeForce(real x, real y, real z,
		   real rsmoothing, real mass) {
  
  Force Force;

  /* The trick below, which uses VxMed as a 2D temporary array,
     amounts to subtracting the azimuthally averaged density prior to
     the torque evaluation. This has no impact on the torque, but has
     on the angular speed of the planet and is required for a proper
     location of resonances in a non self-gravitating disk. See
     Baruteau & Masset 2008, ApJ, 678, 483 (arXiv:0801.4413) for
     details. */
#ifdef BM08
  ComputeVmed (Total_Density);
  ChangeFrame (-1, Total_Density, VxMed);
#endif
  /* The density is now the perturbed density */
  FARGO_SAFE(_ComputeForce(x, y, z, rsmoothing, mass)); /* Function/Kernel Launcher. */
  /* We restore the total density below by adding back the azimuthal
     average */
#ifdef BM08
  ChangeFrame (+1, Total_Density, VxMed);
#endif

  
#ifdef FLOAT
  MPI_Allreduce (&localforce, &globalforce, 12, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
#else
  MPI_Allreduce (&localforce, &globalforce, 12, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  
  Force.fx_inner    = globalforce[0];
  Force.fy_inner    = globalforce[1];
  Force.fz_inner    = globalforce[2];
  Force.fx_ex_inner = globalforce[3];
  Force.fy_ex_inner = globalforce[4];
  Force.fz_ex_inner = globalforce[5];
  Force.fx_outer    = globalforce[6];
  Force.fy_outer    = globalforce[7];
  Force.fz_outer    = globalforce[8];
  Force.fx_ex_outer = globalforce[9];
  Force.fy_ex_outer = globalforce[10];
  Force.fz_ex_outer = globalforce[11];

  return Force;

}

Point ComputeAccel(real x, real y, real z,
		   real rsmoothing, real mass) {
  Point acceleration;
  Force force;
  force = ComputeForce (x, y, z, rsmoothing, mass);
  if (EXCLUDEHILL) {
    acceleration.x = force.fx_ex_inner+force.fx_ex_outer;
    acceleration.y = force.fy_ex_inner+force.fy_ex_outer;
    acceleration.z = force.fz_ex_inner+force.fz_ex_outer;
  } 
  else {
    acceleration.x = force.fx_inner+force.fx_outer;
    acceleration.y = force.fy_inner+force.fy_outer;
    acceleration.z = force.fz_inner+force.fz_outer;
  }
  return acceleration;
}

void AdvanceSystemFromDisk(real dt) {
  int NbPlanets, k;
  Point gamma;
  real x, y, z;
  real vx, vy, vz;
  real gammax, gammay, gammaz;
  real r, m, smoothing;
  NbPlanets = Sys->nb;
  for (k = 0; k < NbPlanets; k++) {
    if (Sys->FeelDisk[k] == 1) {
      m = Sys->mass[k];
      x = Sys->x[k];
      y = Sys->y[k];
      z = Sys->z[k];
      r = sqrt(x*x + y*y + z*z);
      if (ROCHESMOOTHING != 0)
	smoothing = r*pow(m/3./MSTAR,1./3.)*ROCHESMOOTHING;
      else
	smoothing = ASPECTRATIO*pow(r/R0,FLARINGINDEX)*r*THICKNESSSMOOTHING;
      gamma = ComputeAccel (x, y, z, smoothing, m);
      Sys->vx[k] += dt * gamma.x;
      Sys->vy[k] += dt * gamma.y;
      Sys->vz[k] += dt * gamma.z;
#ifdef GASINDIRECTTERM
      Sys->vx[k] += dt * IndirectTerm.x;
      Sys->vy[k] += dt * IndirectTerm.y;
      Sys->vz[k] += dt * IndirectTerm.z;
#endif
	} else if (Sys->FeelDisk[k] == 2) {
      m = Sys->mass[k];
      x = Sys->x[k];
      y = Sys->y[k];
      z = Sys->z[k];
      vx = Sys->vx[k];
      vy = Sys->vy[k];
      vz = Sys->vz[k];
      r = sqrt(x*x + y*y + z*z);
      if (ROCHESMOOTHING != 0)
        smoothing = r*pow(m/3./MSTAR,1./3.)*ROCHESMOOTHING;
      else
        smoothing = ASPECTRATIO*pow(r/R0,FLARINGINDEX)*r*THICKNESSSMOOTHING;
      //gamma = ComputeAccel (x, y, z, smoothing, m);
		if (Sys->taum[k] != 0.0){
        gammax = -Sys->vx[k]/Sys->taum[k];
        gammay = -Sys->vy[k]/Sys->taum[k];
        gammaz = -Sys->vz[k]/Sys->taum[k];
      }
      if (Sys->taue[k] != 0.0){
        const double vdotr = x*vx + y*vy + z*vz;
        const double prefac = -2*vdotr/r/r/Sys->taue[k];
        gammax += prefac*x;
        gammay += prefac*y;
        gammaz += prefac*z;
      }
	  Sys->vx[k] += dt * gammax;
      Sys->vy[k] += dt * gammay;
      Sys->vz[k] += dt * gammaz;
#ifdef GASINDIRECTTERM
      Sys->vx[k] += dt * IndirectTerm.x;
      Sys->vy[k] += dt * IndirectTerm.y;
      Sys->vz[k] += dt * IndirectTerm.z;
#endif
    } else if ((Sys->FeelDisk[k] == 0) && (Sys->Flag_Pres[k] == YES)) {
    UpdatePlanetFromTrajectory(Sys, k, PhysicalTime);
	}
  }
}

OrbitalElements SV2OE (StateVector v, real m) {
  real x,y,z,vx,vy,vz;
  real Ax, Ay, Az, h, h2, inc, e;
  real d, hx, hy, hz, a, E, M, V;
  real hhor, per, an;//Ascending node
  OrbitalElements o;
  x = v.x;
  y = v.y;
  z = v.z;
  vx = v.vx;
  vy = v.vy;
  vz = v.vz;

  d = sqrt(x*x+y*y+z*z);
  
  hx   = y*vz - z*vy;
  hy   = z*vx - x*vz;
  hz   = x*vy - y*vx;
  hhor = sqrt(hx*hx + hy*hy);

  h2  = hx*hx + hy*hy + hz*hz;
  h   = sqrt(h2);
  o.i = inc = asin(hhor/h);

  Ax = vy*hz-vz*hy - G*m*x/d; // v x h - ri/abs(r);
  Ay = vz*hx-vx*hz - G*m*y/d;
  Az = vx*hy-vy*hx - G*m*z/d;

  o.e = e = sqrt(Ax*Ax+Ay*Ay+Az*Az)/(G*m); //Laplace-Runge-Lenz vector
  o.a = a = h*h/(G*m*(1.-e*e));

  //Eccentric anomaly
  if (e != 0.0) {
    E = acos((1.0-d/a)/e); //E evaluated as such is between 0 and PI
  } else {
    E = 0.0;
  }
  if (x*vx+y*vy+z*vz < 0) E= -E; //Planet goes toward central object,
  //hence on its way from aphelion to perihelion (E < 0)

  if (isnan(E)) {
    if (d < a) 
      E = 0.0;
    else
      E = M_PI;
  }

  o.M = M = E-e*sin(E);
  o.E = E;

  //V: true anomaly
  if (e > 1.e-14) {
    V = acos ((a*(1.0-e*e)/d-1.0)/e);
  } else {
    V = 0.0;
  }
  if (E < 0.0) V = -V;

  o.ta = V;
  
  if (fabs(o.i) > 1e-5) {
    an = atan2(hy,hx)+M_PI*.5; //Independently of sign of (hz)
    if (an > 2.0*M_PI) an -= 2.0*M_PI;
  } else {
    an = 0.0;//Line of nodes not determined ==> defaults to x axis
  }

  o.an = an;

  // Argument of periapsis
  per = acos((Ax*cos(an)+Ay*sin(an))/sqrt(Ax*Ax+Ay*Ay+Az*Az));
  if ((-hz*sin(an)*Ax+hz*cos(an)*Ay+(hx*sin(an)-hy*cos(an))*Az) < 0.0)
    per = 2.0*M_PI-per;
  o.per = per;
  if (Ax*Ax+Ay*Ay > 0.0)
    o.Perihelion_Phi = atan2(Ay,Ax);
  else
    o.Perihelion_Phi = atan2(y,x);
  return o;
}

void FindOrbitalElements (StateVector v,real m,int n){
  FILE *output;
  char name[256];
  OrbitalElements o;
  if (CPU_Rank) return;
  sprintf (name, "%sorbit%d.dat", OUTPUTDIR, n);
  output = fopen_prs (name, "a");
  o = SV2OE (v,m);
 
  fprintf (output, "%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g", \
	   PhysicalTime, o.e, o.a, o.M, o.ta, o.per, XAxisRotationAngle);
  fprintf (output, "\t%.12g\t%.12g\t%.12g\n", o.i, o.an, o.Perihelion_Phi);
  fclose (output);
}

void SolveOrbits (PlanetarySystem *sys){
  int i, n;
  StateVector v;
  n = sys->nb;
  for (i = 0; i < n; i++) {
    v.x = sys->x[i];
    v.y = sys->y[i];
    v.z = sys->z[i];
    v.vx = sys->vx[i];
    v.vy = sys->vy[i];
    v.vz = sys->vz[i];
    FindOrbitalElements (v,MSTAR+sys->mass[i],i);
  }
} 
