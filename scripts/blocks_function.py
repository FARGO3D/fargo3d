import re
import getopt
import shlex
import sys
import os
import numpy as np

SETUPDIR   = "setups"
SCRIPTSDIR = "scripts"
SRCDIR     = "src"
BINDIR     = "bin"
C2CUDA     = "scripts/c2cuda.py"
PYTHON_CMD = shlex.quote(sys.executable) + " "
    
def usage():
    print('\n(blocks.py) Usage: -s --setup=  --> SETUP NAME')
    print('         -g --filename= --> CPU/GPU FILE NAME')
    print('         -f --force --> Force new profiling of functions that already exist in the .blocks file.')
    exit()

def opt_reader():
    try:
        options, remainder = getopt.getopt(sys.argv[1:],
                                           's:g:f',
                                           ['setup=',
                                            'filename=',
                                            'force'])
    except getopt.GetoptError:
        usage()

    if(options == []):
        usage()

    force = False
    silent =" > /dev/null 2>&1"
        
    for opt,arg in options:
        if opt in ('-s', '--setup'):
            setup = arg
            continue
        if opt in ('-g', '--filename'):
            filename = arg
            continue
        if opt in ('-f', '--force'):
            force = True
            continue

    global SETUPNAME; SETUPNAME = setup
    global FILENAME; FILENAME = filename
    global FORCE; FORCE = force
    global SILENT; SILENT = silent

def append_blocks(x,y,z):
    try:
        with open(SETUPDIR+'/'+SETUPNAME+'/'+SETUPNAME+".blocks", 'a+', encoding='utf-8') as blocks:
            blocks.seek(0)
            found = False
            for line in blocks:
                new_line = line.split()
                if new_line and new_line[0] == FILENAME[:-2]:
                    found = True
            if not found:
                blocks.write(FILENAME[:-2]+'\t{0:d}\t{1:d}\t{2:d}\n'.format(x,y,z))
                print(FILENAME[:-2]+'\t{0:d}\t{1:d}\t{2:d}'.format(x,y,z) + "\t appended")
            else:
                print(FILENAME[:-2]+'\t{0:d}\t{1:d}\t{2:d}'.format(x,y,z) + "\t not appended")
    except OSError:
        print("(blocks.py): Error opening "+SETUPNAME+".blocks")
        
def analyze_data():
    with open(FILENAME[:-2]+'_blocks.temp', 'r', encoding='utf-8') as temp:
        lines = temp.readlines()
    name = []; nx = []; ny = []; nz = []; time = []
    for line in lines:
        match = re.match("("+BINDIR+"/"+FILENAME+'u'+r")\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+.\d+)",line)
        if match:
            name.append(match.group(1))
            nx.append(int(match.group(2)))
            ny.append(int(match.group(3)))
            nz.append(int(match.group(4)))
            time.append(float(match.group(5)))
    index, = np.where(np.array(time) == np.min(time))[0]
    return nx[index], ny[index], nz[index]

def test_function():

    compile_string = "make GPU=1 MPICUDA=0 PARALLEL=0 DEBUG=0 FULLDEBUG=0 FARGO_DISPLAY=0 PROFILING=1 SETUP="+SETUPNAME+SILENT
    os.system(compile_string)
    os.system(PYTHON_CMD+ 
              C2CUDA+' -i ' + 
              SRCDIR +'/'+FILENAME + ' -o ' + 
              BINDIR+'/'+FILENAME + 'u -p'+SILENT)
    os.system(compile_string)
    os.system("rm "+BINDIR+'/'+FILENAME[:-2]+"_gpu.o")
    os.system("rm -f "+BINDIR+'/'+FILENAME+'u ')
    os.system("./fargo3d -f "+SETUPDIR+'/'+SETUPNAME+'/'+SETUPNAME+'.par > '+ FILENAME[:-2]+'_blocks.temp 2>&1')
    skip = False
    try:
        blocks = analyze_data()
    except:
        print(FILENAME[:-2]+" has been skipped.")
        skip = True
        pass
    os.system("rm -f "+FILENAME[:-2]+"_blocks.temp")
    if skip:
        exit()
    return blocks

opt_reader()
x,y,z = test_function()
append_blocks(x,y,z)
