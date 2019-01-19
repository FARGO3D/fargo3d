import sys
import os
here = os.path.realpath("."); path = sys.path.insert(0,here+"/scripts")
try:
    import test as T
except ImportError:
    print "\nError!!! test module can not be imported. Be sure tht you're executig the test from the main directory, using make for that.\n"

description1 = """Testing a restart with the p3diso setup.
Initial run is on one processor only.\n"""
description2 = """Restarting the simulation with 4 processors.\n"""
RestartTest = T.GenericTest(testname = "RESTART_TESTP3D",
                            flags1 = "SETUP=p3diso  PARALLEL=0 FARGO_DISPLAY=NONE GPU=0",
                            launch1 = "./fargo3d",
                            description1 = description1,
                            flags2 = "SETUP=p3diso PARALLEL=1 FARGO_DISPLAY=NONE GPU=0",
                            launch2 = "mpirun -np 4 ./fargo3d -S 1",
                            description2 = description2,
                            parfile = "setups/p3diso/p3diso.par",
                            verbose = False,
                            plot=False,
                            field = "gasdens",
                            compact = True,
                            parameters = {'dt':0.5, 'ninterm':1 ,'ntot':2,
                                          'nx':80, 'ny':35, 'nz':10},
                            clean = True,
                            n = 2)

RestartTest.set_commands(command1 = "mkdir RESTART_TESTP3D/test2; cp RESTART_TESTP3D/test1/* RESTART_TESTP3D/test2/",
                         command2 = None)
RestartTest.run()
