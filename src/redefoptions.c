#include "fargo3d.h"

HashParam CommandLineParams[100]; /* should be enough... */
long nbparamCL=0;

void OptionError () {
  prs_error ("Incorrect parameter redefinition on command line\n");
}

void ParseRedefinedOptions (char *CommandLineRedefinedOptions) 
{
  char buffer[MAXLINELENGTH];
  char *current, *s1, *s2;
  long length;
  long parity=0, i;
  current = CommandLineRedefinedOptions;
  length = (long)strlen(CommandLineRedefinedOptions);
  if (length > MAXLINELENGTH-1) 
    prs_error ("Very long command line. Please rebuild code after increasing MAXLINELENGTH");
  if (length < 3)
    OptionError ();
  while (current-CommandLineRedefinedOptions < length-1) {
    strcpy (buffer, current);
    s1 = buffer + strspn(buffer, "\t :=>,;"); /* Skip initial separators */
    s2 = s1 + strcspn(s1, "\t :=>,;");	      /* Read until a separator is found  */
    *s2 = 0;				      /* We end the string here */
    current += (s2-buffer); 	/* We move the current position on option line accordingly */
    parity = 1-parity;
    if (parity) {
      for (i = 0; i < strlen(s1); i++)
	s1[i] = (char) toupper(s1[i]);
      strcpy (CommandLineParams[nbparamCL].name, s1);
    }
    else {
      strcpy (CommandLineParams[nbparamCL].stringvalue, s1);
      CommandLineParams[nbparamCL].floatvalue = strtod (s1, NULL);
      CommandLineParams[nbparamCL].intvalue = strtol (s1, NULL, 10);
      CommandLineParams[nbparamCL].boolvalue = FALSE;
      if ((*s1 == 'Y') || (*s1 == 'E') || (*s1 == 'T') || (*s1 == '1') ||\
	  (*s1 == 'y') || (*s1 == 'e') || (*s1 == 't'))
	CommandLineParams[nbparamCL].boolvalue = TRUE;
      nbparamCL++;
    }
  }
  if (parity)
    OptionError ();
}
