import sys
import os
here = os.path.realpath("."); path = sys.path.insert(0,here+"/scripts")
try:
    import test as T
except ImportError:
    print "\nError!!! test module can not be imported. Be sure that you're executing the test from the main directory, using make for that.\n"

description1 = """Running otvortex on the CPU.\n"""
description2 = """Running otvortex on the GPU.\n"""
CPUGPUTest = T.GenericTest(testname = "CPU_vs_GPU_TEST",
                           flags1 = "SETUP=otvortex PARALLEL=0 FARGO_DISPLAY=NONE GPU=0",
                           launch1 = "./fargo3d",
                           description1 = description1,
                           flags2 = "SETUP=otvortex PARALLEL=0 FARGO_DISPLAY=NONE GPU=1",
                           launch2 = "./fargo3d -D 0",
                           description2 = description2,
                           parfile = "setups/otvortex/otvortex.par",
                           verbose = False,
                           plot=False,
                           field = "gasdens",
                           compact = True,
                           parameters = {'dt':0.2, 'ninterm':1, 'ntot':1,
                                         'ny':64, 'nz':64, 'nx':1},
                           clean = True)
CPUGPUTest.run()
