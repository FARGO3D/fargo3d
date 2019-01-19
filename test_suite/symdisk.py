import sys
import os
here = os.path.realpath("."); path = sys.path.insert(0,here+"/scripts")
try:
    import test as T
except ImportError:
    print "\nError!!! test module can not be imported. Be sure that you are executing the test from the main directory, using make for that.\n"

description1 = """Testing a disk setup (half disk up to the equator, versus full disk)
We start by running a half disk (setup p3diso) on 3 processors"""
description2 = """Restarting the simulation with a full disk (setup p3disof) on 4 processors.\n"""
RestartTest = T.GenericTest(testname = "SYMDISK_TEST",
                            flags1 = "SETUP=p3diso  PARALLEL=1 FARGO_DISPLAY=NONE GPU=0",
                            launch1 = "mpirun -np 3 ./fargo3d",
                            description1 = description1,
                            flags2 = "SETUP=p3disof PARALLEL=1 FARGO_DISPLAY=NONE GPU=0",
                            launch2 = "mpirun -quiet -np 4 ./fargo3d",
                            description2 = description2,
                            parfile = "setups/p3diso/p3diso.par",
                            verbose = False,
                            plot=False,
                            field = "gasdens",
                            compact = True,
                            parameters1 = {'dt':0.4, 'ninterm':1 ,'ntot':1,
                                          'nx':80, 'ny':35, 'nz':10},
                            parameters2 = {'dt':0.4, 'ninterm':1 ,'ntot':1,
                                          'nx':80, 'ny':35, 'nz':20,
                                          'zmax': "1.72079632679489661922",
                                          'setup': "p3disof"},
                            clean = True,
                            restore = False)

RestartTest.set_commands(command2 = "head -c 224000 SYMDISK_TEST/test2/gasdens1.dat > temp; /bin/mv -f temp SYMDISK_TEST/test2/gasdens1.dat", command1 = None)
RestartTest.run()
