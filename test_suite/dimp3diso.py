import sys
import os
here = os.path.realpath("."); path = sys.path.insert(0,here+"/scripts")
try:
    import test as T
except ImportError:
    print "\nError!!! test module can not be imported." + \
        "Be sure that you are executing the test from the main directory\n" +\
        "using 'make testmpi'.\n"


description1 = """
Building the scale free parallel version of the setup p3diso
and running it with two processors. 
"""

description2 = """
Building the MKS parallel version of the setup p3diso
and running it with four processors. 
"""

flags1 = "SETUP=p3diso PARALLEL=1 FARGO_DISPLAY=NONE GPU=0 RESCALE=0 UNITS=0"
flags2 = "SETUP=p3diso PARALLEL=1 FARGO_DISPLAY=NONE GPU=0 RESCALE=1 UNITS=MKS"
DimTestp3d = T.GenericTest(testname = "P3DISO_TEST",
                        flags1 = flags1,
                        launch1 = "mpirun -np 2 ./fargo3d -m",
                        description1 = description1,
                        flags2 = flags2,
                        launch2 = "mpirun -np 4 ./fargo3d -m",
                        description2 = description2,
                        parfile = "setups/p3diso/p3diso.par",
                        verbose = False,
                        clean=True,
                        plot=False,
                        field = "gasdens",
                        compact = True,
                        parameters = {'dt':0.8, 'ninterm':1, 'ntot':1,
                                      'ny':20, 'nz':8, 'nx':30})
DimTestp3d.run()
