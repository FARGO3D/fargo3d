from __future__ import print_function
import sys
import os
here = os.path.realpath("."); path = sys.path.insert(0,here+"/scripts")
try:
    import test as T
except ImportError:
    print("\nError!!! test module can not be imported." +
          "Be sure tht you're executig the test from the main directory, using make for that.\n")


flags = "SETUP=otvortex PARALLEL=0 FARGO_DISPLAY=NONE GPU=0"

description1 = """
Preparing the setup Orszag-Tang Vortex in the YZ plane, with one CPU.
"""
description2 = """
Preparing the setup Orszag-Tang Vortex in the XZ plane, with one CPU.
"""
OTTest = T.GenericTest(testname = "OT_VORTEX_YZ-XZ",
                       flags1 = flags,
                       launch1 = "./fargo3d",
                       description1 = description1,
                       flags2 = flags,
                       launch2 = "./fargo3d",
                       description2 = description2,
                       parfile = "setups/otvortex/otvortex.par",
                       verbose = False,
                       plot=False,
                       field = "gasdens",
                       compact = True,
                       parameters1 = {'dt':0.01, 'ninterm':1, 'ntot':1,
                                      'ny':128, 'nz':128, 'nx':1},
                       parameters2 = {'dt':0.01, 'ninterm':1, 'ntot':1,
                                      'ny':1, 'nz':128, 'nx':128},
                       parameters = None,
                       clean = True,
                       keep = True,
                       restore = False)

OTTest.run()
flag1 = OTTest.get_status()

description1 = """
Preparing the setup Orszag-Tang Vortex in the YZ plane, with one CPU.
"""
description2 = """
Preparing the setup Orszag-Tang Vortex in the XY plane, with one CPU.
"""
OTTest = T.GenericTest(testname = "OT_VORTEX_YZ-XY",
                       flags1 = flags,
                       launch1 = "./fargo3d",
                       description1 = description1,
                       flags2 = flags,
                       launch2 = "./fargo3d",
                       description2 = description2,
                       parfile = "setups/otvortex/otvortex.par",
                       verbose = False,
                       plot=False,
                       field = "gasdens",
                       compact = True,
                       parameters1 = {'dt':0.01, 'ninterm':1, 'ntot':1,
                                      'ny':128, 'nz':128, 'nx':1},
                       parameters2 = {'dt':0.01, 'ninterm':1, 'ntot':1,
                                      'ny':128, 'nz':1, 'nx':128},
                       parameters = None,
                       clean = True,
                       restore = False,
                       keep = False)

OTTest.run()
flag2 = OTTest.get_status()

description1 = """
Preparing the setup Orszag-Tang Vortex in the XY plane, with one CPU.
"""
description2 = """
Preparing the setup Orszag-Tang Vortex in the XZ plane, with one CPU.
"""
OTTest = T.GenericTest(testname = "OT_VORTEX_XY-XZ",
                       flags1 = flags,
                       launch1 = "./fargo3d",
                       description1 = description1,
                       flags2 = flags,
                       launch2 = "./fargo3d",
                       description2 = description2,
                       parfile = "setups/otvortex/otvortex.par",
                       verbose = False,
                       plot=False,
                       field = "gasdens",
                       compact = True,
                       parameters1 = {'dt':0.01, 'ninterm':1, 'ntot':1,
                                      'ny':128, 'nz':1, 'nx':128},
                       parameters2 = {'dt':0.01, 'ninterm':1, 'ntot':1,
                                      'ny':1, 'nz':128, 'nx':128},
                       parameters = None,
                       clean = True,
                       restore = True,
                       keep = False)

OTTest.run()
flag3 = OTTest.get_status()

if flag1 and flag2 and flag3:
    print("\n======================================================")
    print("Test of permutations of Orzsag-Tang Vortex was passed.")
    print("======================================================\n")
else:
    print("\n===================================================")
    print("Test of Orzsag-Tang Vortex With one GPU was failed.")
    print("===================================================\n")
