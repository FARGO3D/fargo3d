#include <Python.h>
#include "fargo3d.h"

#define YZSIM 1
#define XZSIM 2
#define XYSIM 3

void pyrun(const char *command, ...) {
  va_list list;
  char string[512];
  va_start(list, command);
  vsprintf(string, command, list);
  va_end(list);
  PyRun_SimpleString(string);
}

void finalize_python() {
  Py_Exit(0);
}

int check_simtype() {
  if(Nz == 1)
    return XYSIM;
  if(Ny == 1)
    return XZSIM;
  if(Nx == 1)
    return YZSIM;
  printf("Fatal error in matplotlib.c. Check the function check_simtype()\n");
    return -1;
}

void initialize_python() {
  
  Py_InitializeEx(1);
  //Bug with python event handler!!!! http://www.vtk.org/Bug/view.php?id=13788
  pyrun("import signal;" 
	"signal.signal(signal.SIGINT, signal.SIG_DFL);");
  pyrun("import matplotlib");
  pyrun("matplotlib.use('TKAgg')");
  pyrun("import matplotlib.pyplot as plt");
  pyrun("import matplotlib as mpl");
  pyrun("import numpy as np");
  pyrun("mpl.rcParams['toolbar'] = 'None'");
  pyrun("plt.ion()");
  pyrun("fig = plt.figure('FARGO3D - %02d')", CPU_Rank);

}

void plot1d(char* name, int n, int merge) {
  static int init = 0;
  int dim;
  real x1, x2, y1, y2;
  if (merge) {
    if(NX != 1) {
      dim = NX;
      x1 = XMIN; x2 = XMAX;
    }
    if(NY != 1) {
      dim = NY;
      x1 = YMIN; x2 = YMAX;
    }
    if(NZ != 1) {
      dim = NZ;
      x1 = ZMIN; x2 = ZMAX;
    }
  }
  else {
    if(NX != 1) {
      dim = Nx;
      x1 = XMIN; x2 = XMAX;
    }
    if(NY != 1) {
      dim = Ny;
      x1 = Ymin(NGHY); x2 = Ymin(Ny+NGHY);
    }
    if(NZ != 1) {
      dim = Nz;
      x1 = Zmin(NGHZ); x2 = Zmin(Nz+NGHZ);
    }
  }

  if (init == 0)
    initialize_python();

  if (merge){
#ifndef FLOAT
    pyrun("field = np.fromfile('%s%s%d.dat')",OUTPUTDIR, name, n);
#else
    pyrun("field = np.fromfile('%s%s%d.dat',dtype='f32')",OUTPUTDIR, name, n);
#endif
  }
  else {
    pyrun("field = np.fromfile('%s%s%d_%d.dat')",OUTPUTDIR, name, n, CPU_Rank);
  }

  if(PLOTLOG) {
    pyrun("field = np.log10(field)");
  }

  if (init == 0) {
    pyrun("ax = fig.add_subplot(111,axisbg='k')");
    pyrun("ax.set_title('Field: %s',color='w')",name);  

    pyrun("ax.set_ylabel('%s')",name);

#ifdef X
    pyrun("ax.set_xlabel('X')");
#endif
#ifdef Y
    pyrun("ax.set_xlabel('Y')");
#endif
#ifdef Z
    pyrun("ax.set_xlabel('Z')");
#endif
#ifdef FLOAT
    pyrun("domain = np.linspace(%f,%f,%d)",x1,x2,dim);
#else
    pyrun("domain = np.linspace(%lf,%lf,%d)",x1,x2,dim);
#endif
    pyrun("time = ax.text(0.98, 0.97,'Time: %3.5f',"
	  "horizontalalignment='right',"
	  "verticalalignment='center',"
	  "transform = ax.transAxes, color='w')", PhysicalTime);
    pyrun("output = ax.text(0.02, 0.97,'Output: %03d',"
	  "horizontalalignment='left',"
	  "verticalalignment='center',"
	  "transform = ax.transAxes, color='w')", n);
    pyrun("plot = plt.plot(domain,field,'w-',linewidth=2)");

    if(AUTOCOLOR) {
      pyrun("ax.relim()");
    }
    else {
#ifdef FLOAT
      pyrun("ax.set_ylim([%f,%f])",VMIN,VMAX);
#else
      pyrun("ax.set_ylim([%lf,%lf])",VMIN,VMAX);
#endif
    }

    pyrun("fig.canvas.flush_events()");
    init = 1;
  }
  else {
    pyrun("time.set_text('Time: %3.5f')", PhysicalTime);
    pyrun("output.set_text('Output: %03d')", n);

    pyrun("plot[0].set_data(domain,field)");
    if(AUTOCOLOR) {
      pyrun("ax.relim()");
    }
    else {
#ifdef FLOAT
      pyrun("ax.set_ylim([%f,%f])",VMIN,VMAX);
#else
      pyrun("ax.set_ylim([%lf,%lf])",VMIN,VMAX);
#endif
    }
    pyrun("fig.canvas.flush_events()");
  }
}

void plot2d(char* name, int n, int merge) {

  static int init = 0;
  int n1, n2;
  real x1, x2, y1, y2;

  if (merge) {
    if(NX == 1) {
      n1 = NY;  n2 = NZ;
      x1 = YMIN; x2 = YMAX;
      y1 = ZMIN; y2 = ZMAX;
    }
    if(NY == 1) {
      n1 = NX;  n2 = NZ;
      x1 = XMIN; x2 = XMAX;
      y1 = ZMAX; y2 = ZMIN;
    }
    if(NZ == 1) {
      n1 = NX;  n2 = NY;
      x1 = XMIN; x2 = XMAX;
      y1 = YMIN; y2 = YMAX;
    }
  }
  else {
    if(NX == 1) {
      n1 = Ny;  n2 = Nz;
      x1 = Ymin(NGHY); x2 = Ymin(Ny+NGHY);
      y1 = Zmin(NGHY); y2 = Zmin(Nz+NGHZ);
    }
    if(NY == 1) {
      n1 = Nx;  n2 = Nz;
      x1 = XMIN; x2 = XMAX;
      y1 = Zmin(NGHZ); y2 = Zmin(Nz+NGHZ);
    }
    if(NZ == 1) {
      n1 = Nx;  n2 = Ny;
      x1 = XMIN; x2 = XMAX;
      y1 = Ymin(NGHY); y2 = Ymin(Ny+NGHY);
    }
  }
  
  if (init == 0)
    initialize_python();

#ifdef FLOAT
  pyrun("dtype = np.float32");
#else
  pyrun("dtype = np.float64");
#endif
  if (merge) {
    if (VTK)
      pyrun("f = open('%s%s%d.vtk')",OUTPUTDIR, name, n);
    else {
      pyrun("f = open('%s%s%d.dat')",OUTPUTDIR, name, n);
    }
  }
  else {
    if (VTK)
      pyrun("f = open('%s%s%d_%d.vtk')",OUTPUTDIR, name, n,CPU_Rank);
    else
      pyrun("f = open('%s%s%d_%d.dat')",OUTPUTDIR, name, n,CPU_Rank);
  }

  if (VTK) {
    pyrun("f.seek(%d,0)",VtkPosition);
    if (NX == 1)
      pyrun("field = np.fromfile(f,dtype=dtype).newbyteorder().reshape([%d,%d])", n2,n1);
    else {
      pyrun("field = np.swapaxes(np.fromfile(f,dtype=dtype).newbyteorder().reshape([%d,%d]),0,1)", n1, n2);
    }
  }
  else
    pyrun("field = np.fromfile(f,dtype=dtype).reshape([%d,%d])",n2, n1);
  
  pyrun("f.close()");
  
  if(PLOTLOG) {
    pyrun("field = np.log10(field)");
  }

  if (init == 0) {
    pyrun("ax = fig.add_subplot(111)");
    pyrun("ax.set_title('Field: %s',color='w')",name);
    
    int simtype = check_simtype();

    if (simtype == YZSIM) {
      pyrun("ax.set_xlabel('Y')");
      pyrun("ax.set_ylabel('Z')");
    }
    if (simtype == XZSIM) {
      pyrun("ax.set_xlabel('X')");
      pyrun("ax.set_ylabel('Z')");
    }
    if (simtype == XYSIM) {
      pyrun("ax.set_xlabel('X')");
      pyrun("ax.set_ylabel('Y')");
    }
    
    pyrun("time = ax.text(0.98, 0.97,'Time: %3.5f',"
	  "horizontalalignment='right',"
	  "verticalalignment='center',"
	  "transform = ax.transAxes, color='w')", PhysicalTime);
    pyrun("output = ax.text(0.02, 0.97,'Output: %03d',"
	  "horizontalalignment='left',"
	  "verticalalignment='center',"
	  "transform = ax.transAxes, color='w')", n);
    
    pyrun("image = plt.imshow(field,"
	  "cmap = plt.cm.%s,"
	  "extent = [%lf,%lf,%lf,%lf],"
	  "aspect = '%s', origin='lower')",
	  CMAP,x1,x2,y1,y2,ASPECT);

    if(COLORBAR) {
      pyrun("plt.colorbar()");
    }
    pyrun("fig.canvas.flush_events()");
    init = 1;
  }
  else {
    pyrun("time.set_text('Time: %3.5f')", PhysicalTime);
    pyrun("output.set_text('Output: %03d')", n);
    pyrun("image.set_data(field)");
    if(AUTOCOLOR) {
      pyrun("image.autoscale()");
    }
    else {
      pyrun("image.set_clim(%lf,%lf)",VMIN,VMAX);
    }
    
    pyrun("fig.canvas.flush_events()");
  }
}

void plot3d(char* name, int n, int merge) {
  static boolean init = YES;  
  int n1,n2,n3;

  if (init)
    initialize_python();
    pyrun("ax = fig.add_subplot(111)");
    pyrun("ax.set_title('Field: %s',color='w')",name);
    
  if (merge) {
    n1 = NX;  n2 = NY; n3 = NZ;
  }
  else {
    n1 = Nx;  n2 = Ny; n3 = Nz;
  }

#ifdef FLOAT
  pyrun("dtype = np.float32");
#else
  pyrun("dtype = np.float64");
#endif
  if (merge) {
    if (VTK)
      pyrun("f = open('%s%s%d.vtk')",OUTPUTDIR, name, n);
    else
      pyrun("f = open('%s%s%d.dat')",OUTPUTDIR, name, n);
  }
  else {
    if (VTK)
      pyrun("f = open('%s%s%d_%d.vtk')",OUTPUTDIR, name, n,CPU_Rank);
    else
      pyrun("f = open('%s%s%d_%d.dat')",OUTPUTDIR, name, n,CPU_Rank);
  }
  if (VTK) {
    pyrun("f.seek(%d,0)",VtkPosition);
    pyrun("field = np.fromfile(f,dtype=dtype).newbyteorder().reshape([%d,%d,%d])", n3,n2,n1);
  }
  else
    pyrun("field = np.fromfile(f,dtype=dtype).reshape([%d,%d,%d])", n3, n2, n1);

  pyrun("f.close()");
  
  if(PLOTLOG) {
    pyrun("field = np.log10(field)");
  }
  
  if (init) {
    pyrun("time = ax.text(0.98, 0.97,'Time: %3.5f',"
	  "horizontalalignment='right',"
	  "verticalalignment='center',"
	  "transform = ax.transAxes, color='w')", PhysicalTime);
    pyrun("output = ax.text(0.02, 0.97,'Output: %03d',"
	  "horizontalalignment='left',"
	  "verticalalignment='center',"
	  "transform = ax.transAxes, color='w')", n);

    pyrun("image = plt.imshow(%s,"
	  "cmap = plt.cm.%s,"
	  "aspect = '%s')",
	  PLOTLINE,CMAP,ASPECT);

    if(COLORBAR) {
      pyrun("plt.colorbar()");
    }
    pyrun("fig.canvas.flush_events()");
    init = NO;
  }
  else {
    pyrun("time.set_text('Time: %3.5f')", PhysicalTime);
    pyrun("output.set_text('Output: %03d')", n);
    pyrun("image.set_data(%s)",PLOTLINE);
    if(AUTOCOLOR) {
      pyrun("image.autoscale()");
    }
    else {
      pyrun("image.set_clim(%lf,%lf)",VMIN,VMAX);
    }
    pyrun("fig.canvas.flush_events()");
 }
}

void Display() {
  
#if ((defined(X) && defined(Y) && !defined(Z)) || \
     (defined(X) && defined(Z) && !defined(Y)) || \
     (defined(Y) && defined(Z) && !defined(X)))
  if (Merge) {
    if (CPU_Master) {
      plot2d(FIELD, TimeStep, Merge);
    }
  }
  else {
    plot2d(FIELD, TimeStep, Merge);
  }
#endif

#if (defined(X) && defined(Y) && defined(Z))
  if (Merge) {
    if (CPU_Master) {
      plot3d(FIELD, TimeStep, Merge);
    }
  }
  else {
    plot3d(FIELD, TimeStep, Merge);
  }
#endif
  
#if ((defined(X) & !(defined(Y) || defined(Z))) ||  \
     (defined(Y) & !(defined(X) || defined(Z)))  || \
     (defined(Z) & !(defined(X) || defined(Y))))
  plot1d(FIELD, TimeStep, Merge);
#endif
}
