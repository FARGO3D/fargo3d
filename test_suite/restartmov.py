# Test used for backward compatibility nightly built
import sys
import os
here = os.path.realpath("."); path = sys.path.insert(0,here+"/scripts")
try:
    import test as T
except ImportError:
    print "\nError!!! test module can not be imported. Be sure that you're executing the test from the main directory, using make for that.\n"

description1 = """Testing a restart with the p3diso setup (including Stockholm's damping, a mobile massive planet and the indirect term)
Initial run is on 2 processors.\n"""
description2 = """Restarting the simulation with 4 processors.\n"""
RestartTest = T.GenericTest(testname = "RESTART_TESTMOV",
                            flags1 = "SETUP=p3diso PARALLEL=1 FARGO_DISPLAY=NONE GPU=0",
                            launch1 = "mpirun -np 2 ./fargo3d",
                            description1 = description1,
                            flags2 = "SETUP=p3diso PARALLEL=1 FARGO_DISPLAY=NONE GPU=0",
                            launch2 = "mpirun -quiet -np 4 ./fargo3d -S 1",
                            description2 = description2,
                            parfile = "setups/p3diso/p3diso.par",
                            verbose = False,
                            plot=False,
                            field = "gasdens",
                            compact = True,
                            parameters = {'dt':0.4, 'ninterm':2 ,'ntot':5,
                                          'nx':80, 'ny':35, 'nz':10, 'sigma0':3.1e-3,
                                          'PlanetConfig':"planets/MobileJupiter.cfg",
                                          'IndirectTerm':"YES"},
                            clean = False,
                            restore = False,
                            n = 2)

RestartTest.set_commands(command1 = "mkdir RESTART_TESTMOV/test2; cp RESTART_TESTMOV/test1/* RESTART_TESTMOV/test2/",
                         command2 = None)
RestartTest.run()
