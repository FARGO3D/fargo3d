import sys
import os
here = os.path.realpath("."); path = sys.path.insert(0,here+"/scripts")
try:
    import test as T
except ImportError:
    print "\nError!!! test module cannot be imported." + \
        "Be sure that you are executing the test from the main directory\n" +\
        "using 'make testmpi'.\n"


##### IMPORTANT NOTE ############
# IN ORDER TO PASS THIS TEST
# (make testdim in main directory)
# you must edit setups/mri/mri.opt
# as follows:
# 1. Comment out FARGO_OPT += -DFLOAT
# 2. Uncomment FARGO_OPT += -DSTRICTSYM
#################################
        

cmd = os.popen('grep STRICTSYM setups/mri/mri.opt   | grep -c "#"')
c = cmd.read()
cmd.close()
if (int(c) > 0):
    print "\033[1m\033[31mPLEASE READ THE IMPORTANT NOTE IN FILE "+__file__+"\033[0m"
    print "Take action and retry"
    exit ()

description1 = """
Building the scale free version of the setup mri
and running it on one processor. 
"""

description2 = """
Building the MKS version of the setup mri
and running it on one processor. 
"""

flags1 = "SETUP=mri PARALLEL=0 FARGO_DISPLAY=NONE GPU=0 RESCALE=0 UNITS=0"
flags2 = "SETUP=mri PARALLEL=0 FARGO_DISPLAY=NONE GPU=0 RESCALE=1 UNITS=MKS"
DimTest = T.GenericTest(testname = "DIM_TEST",
                        flags1 = flags1,
                        launch1 = "./fargo3d",
                        description1 = description1,
                        flags2 = flags2,
                        launch2 = "./fargo3d",
                        description2 = description2,
                        parfile = "setups/mri/mri.par",
                        verbose = False,
                        clean=True,
                        plot=False,
                        restore=False,
                        field = "gasdens",
                        compact = True,
                        parameters = {'dt':0.4, 'ninterm':1, 'ntot':1,
                                      'nx':60, 'ny':40, 'nz':12, 'eta':1e-4})
DimTest.run()
