#include "fargo3d.h"

/* In this file we read the default root directory for output.  They
 should be defined by the environment variable FARGO_OUT. If this
 variable is undefined, it is set by default to ./ */

void ReadDefaultOut () {
  char *defout;
  defout = getenv ("FARGO_OUT");
  if (defout != NULL) {
    strncpy (DefaultOut, defout, MAXLINELENGTH-2);
  } else {
    sprintf (DefaultOut, "./");
  }
  if (*(DefaultOut+strlen(DefaultOut)-1) != '/')
    strcat (DefaultOut, "/");	/* Add trailing slash if missing */
  masterprint ("The default output directory root is %s\n", DefaultOut);
}

void SubsDef (target, def)
  char *target, *def;
{
  char c='@';
  char new_target[MAXLINELENGTH];
  char *loc, *follow;
  loc = strchr (target, (int)c);
  if (loc != NULL) {
    if (*(loc+1) == '/')
      follow=loc+2;
    else
      follow=loc+1;
    if (loc != target) {
      masterprint ("Characters located before '%c' wildcard in OUTPUTDIR definition are ignored\n", c);
    }
    snprintf (new_target,MAXLINELENGTH/2-1,"%s",def);
    strncat (new_target,follow,MAXLINELENGTH/2-1);
    strncpy (target, new_target, MAXLINELENGTH-1);
  }
  masterprint ("The output directory is %s\n", target);
}
