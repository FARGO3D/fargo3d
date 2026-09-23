#include "fargo3d.h"

#define WALLCLOCK 1

#ifdef WALLCLOCK
#define TIMETICKS wcticks
#else
#define TIMETICKS buffer.tms_utime
#endif

static clock_t  First, Preceeding, Current, FirstUser, CurrentUser, PreceedingUser;
static long     Ticks;
extern int begin_i;

void InitSpecificTime (process_name, title)
     TimeProcess *process_name;
     char *title;
{
#ifdef PROFILING
  struct tms buffer;
  clock_t wcticks;
  Ticks = sysconf (_SC_CLK_TCK);
  wcticks = times (&buffer);
  process_name->clicks = TIMETICKS;
  strcpy (process_name->name, title);
  MPI_Barrier (MPI_COMM_WORLD);
#endif
}

real GiveSpecificTime (process_name)
     TimeProcess process_name;
{
  real t=0.0;
#ifdef PROFILING
  struct tms buffer;
  clock_t wcticks;
  long ticks;
  MPI_Barrier (MPI_COMM_WORLD);
  Ticks = sysconf (_SC_CLK_TCK);
  wcticks = times (&buffer);
  ticks = TIMETICKS - process_name.clicks;
  t = (real)ticks / (real)Ticks;
  if (process_name.name[0] != 0) {
#ifdef WALLCLOCK
    masterprint ("Wall clock time elapsed during %s : %.3f s\n", process_name.name, t);
#else
    fprintf (stderr, "Time spent in %s : %.3f s\n", process_name.name, t);
#endif
  }
#endif
  return t;
}

void GiveTimeInfo (number)
     int number;
{
  struct tms buffer;
  real total, last, mean, totalu;
  static boolean FirstStep = YES;
  Current = times (&buffer);
  CurrentUser = buffer.tms_utime;
  if (FirstStep == YES) {
    First = Current;
    FirstUser = CurrentUser;
    fprintf (stderr, "Time counters initialized\n");
    FirstStep = NO;
    Ticks = sysconf (_SC_CLK_TCK);
  }
  else {
    total = (real)(Current - First)/Ticks;
    totalu= (real)(CurrentUser-FirstUser)/Ticks;
    last  = (real)(CurrentUser - PreceedingUser)/Ticks;
    number -= begin_i/NINTERM;
    mean  = totalu / number;
    fprintf (stderr, "Total Real Time elapsed    : %.3f s\n", total);
    fprintf (stderr, "Total CPU Time of process  : %.3f s (%.1f %%)\n", totalu, 100.*totalu/total);
    fprintf (stderr, "CPU Time since last time step : %.3f s\n", last);
    fprintf (stderr, "Mean CPU Time between time steps : %.3f s\n", mean);
    fprintf (stderr, "CPU Load on last time step : %.1f %% \n", (real)(CurrentUser-PreceedingUser)/(real)(Current-Preceeding)*100.);
  }     
  PreceedingUser = CurrentUser;
  Preceeding = Current;
}
