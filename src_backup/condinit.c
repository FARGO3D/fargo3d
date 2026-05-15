#include "fargo3d.h"

void CondInit () {
  mastererr ("\n\n******\n\nERROR\n\n");
  mastererr ("You have forgotten to specify initial conditions.\n");
  mastererr ("These must be defined in the setup directory, in a\n");
  mastererr ("file called condinit.c, with a function void CondInit().\n");
  mastererr ("\nOnce this is done, issue a 'make clean' before rebuilding.\n");
  prs_exit (1);
}

