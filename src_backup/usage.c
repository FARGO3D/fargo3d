#include "fargo3d.h"

void PrintUsage (char *execname)
{
  fprintf (stderr, "Usage : %s [-tfCmkpVB0] [-D number] [-s number] [-S number] [+S number] [+/-# arraynb] [+D device_file] parameter_file\n", execname);
  fprintf (stderr, "-C : Force execution of all functions on CPU for a GPU built\n");
  fprintf (stderr, "-D : Manually select the GPU device on which you want to run\n");
  fprintf (stderr, "-m : Merge output files from different processes\n");
  fprintf (stderr, "-k : Do NOT merge output files from different processes\n");
  fprintf (stderr, "-o : redefine parameters on the command line\n");
  fprintf (stderr, "-s : Restart simulation, from output #number (split output)\n");
  fprintf (stderr, "-S : Restart simulation, from output #number (merged output)\n");
  fprintf (stderr, "+S : Overwrite initial conditions with stretched data from output #number\n");
  fprintf (stderr, "-V : Convert DAT output to VTK output\n");
  fprintf (stderr, "-B : Convert VTK output to DAT output\n");
  fprintf (stderr, "-t : Monitor CPU time usage at each time step\n");
  fprintf (stderr, "-# : Give (positive) array number to the code, which can be used as random seed and/or output suffix\n");
  fprintf (stderr, "+# : Give (positive) array number to the code, which can be used as random seed and/or output suffix (early rename)\n");
  fprintf (stderr, "+D : specify a node file in which each line specifies a hostname and a device number\n");
  fprintf (stderr, "-0 : only write initial (or restart) output, and exits,\n");
  fprintf (stderr, "-f : executes only one elementary timestep and exits.\n");
  prs_exit (1);
}

