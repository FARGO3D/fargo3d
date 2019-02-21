from __future__ import print_function
import sys
import os
here = os.path.realpath("."); path = sys.path.insert(0,here+"/scripts")
try:
    import test as T
except ImportError:
    print("\nError!!! test module cannot be imported." + \
          "Be sure that you are executing the test from the main directory\n" +\
          "using 'make testmpi'.\n")


description1 = """
Building the scale free version of the setup fargo
and running it on two GPUS, with forced migration of the planet.
"""

description2 = """
Building the CGS version of the setup fargo and running it on three CPUS,
with forced migration of the planet. 
"""

flags1 = "SETUP=fargo PARALLEL=1 MPICUDA=0 FARGO_DISPLAY=NONE GPU=1 RESCALE=0 UNITS=0"
flags2 = "SETUP=fargo PARALLEL=1 MPICUDA=0 FARGO_DISPLAY=NONE GPU=0 RESCALE=1 UNITS=CGS"
DimTest = T.GenericTest(testname = "DIM_TEST",
                        flags1 = flags1,
                        launch1 = "mpirun -np 2 ./fargo3d -D 0",
                        description1 = description1,
                        flags2 = flags2,
                        launch2 = "mpirun -np 3 ./fargo3d",
                        description2 = description2,
                        parfile = "setups/fargo/fargo.par",
                        verbose = False,
                        clean=True,
                        plot=False,
                        restore=False,
                        field = "gasdens",
                        compact = True,
                        parameters = {'dt':1.0, 'ninterm':10, 'ntot':20,
                                      'nx':64, 'ny':64, 'releasedate':10,
                                      'releaseradius':1.1, 'semimajoraxis':0.9})
DimTest.run()
