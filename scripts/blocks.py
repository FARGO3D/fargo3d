import re
import getopt
import sys
import os
import numpy as np

SETUPDIR   = "setups"
SCRIPTSDIR = "scripts"
SRCDIR     = "src"
BINDIR     = "bin"
C2CUDA     = "scripts/c2cuda.py"

    
def usage():
    print '\n(blocks.py) Usage: -s --setup  --> SETUP NAME'
    print '                   -f --force  --> Force new profiling of functions that already exist in the .blocks file.'
    print
    exit()

def opt_reader():
    try:
        options, remainder = getopt.getopt(sys.argv[1:],
                                           's:f',
                                           ['setup=',
                                            'force'])
    except getopt.GetoptError, err:
        print str(err)
        usage()

    if(options == []):
        usage()

    force = False
    silent =" > /dev/null"
        
    for opt,arg in options:
        if opt in ('-s', '--setup'):
            setup = arg
            continue
        if opt in ('-f', '--force'):
            force = True
            continue

    global SETUPNAME; SETUPNAME = setup
    global FORCE; FORCE = force

def analyze_makefile():
    makefile = open(SRCDIR+"/makefile","r")
    gpu_lines = []
    found = False
    for line in makefile.readlines():
        if line[0:6] == "#<CUDA":
            found = True
            continue
        if line[0:6] == "#</CUD":
            found = False
            continue
        if found:
            gpu_lines.append(line)

    gpu_objects = []

    for obj in gpu_lines:
        obj = obj.split()
        for o in obj:
            if o[-1] == 'o':
                gpu_objects.append(o[:-6]+'.c')
    makefile.close()
    return gpu_objects


opt_reader()
gpu_objects = analyze_makefile()
for name in gpu_objects:
    instruction = "python "+ SCRIPTSDIR + "/blocks_function.py -s "+ SETUPNAME + " -g " + name
    if FORCE:
        instruction += " -f"
    os.system(instruction)
