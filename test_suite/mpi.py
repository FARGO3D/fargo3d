# Test used for backward compatibility nightly built
from __future__ import print_function
import sys
import os
here = os.path.realpath("."); path = sys.path.insert(0,here+"/scripts")
try:
    import test as T
except ImportError:
    print("\nError!!! test module can not be imported." +
          "Be sure that you are executing the test from the main directory\n" +
          "using 'make testmpi'.\n")


description1 = """
Building the parallel version of the setup Orszag-Tang Vortex,
and running it with one processor. 
"""

description2 = """
Running the setup Orszag-Tang Vortex with four processors.
"""

flags = "SETUP=otvortex PARALLEL=1 FARGO_DISPLAY=NONE GPU=0"
MpiTest = T.GenericTest(testname = "MPI_TEST",
                        flags1 = flags,
                        launch1 = "./fargo3d",
                        description1 = description1,
                        flags2 = flags,
                        launch2 = "mpirun -np 4 ./fargo3d",
                        description2 = description2,
                        parfile = "setups/otvortex/otvortex.par",
                        verbose = False,
                        clean=True,
                        plot=False,
                        field = "gasdens",
                        compact = True,
                        parameters = {'dt':0.2, 'ninterm':1, 'ntot':1,
                                      'ny':64, 'nz':64, 'nx':1})
MpiTest.run()
