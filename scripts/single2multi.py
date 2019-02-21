from __future__ import print_function
import re
import shutil
import sys

newlines = """
#FARGO3D 2.0 (multifluid)
#-------------------------------
FLUIDS := 0
NFLUIDS = 1
FARGO_OPT += -DNFLUIDS=${NFLUIDS}
#--------------------------------
"""

newcilines = """
void CondInit() {
   Fluids[0] = CreateFluid("gas",GAS);
   SelectFluid(0);
   Init();
}
"""

try:
    setup = sys.argv[1]
    setupdir = "../setups/"+setup
    shutil.os.chdir(setupdir)
except IndexError:
    print("Usage: python single2multi.py setupname")
    exit()
    
def backup():
    fileslist = shutil.os.listdir(".")
    shutil.os.mkdir("backup")
    for f in fileslist:
        shutil.copy(f,"backup")

backup()
shutil.move(setup+".bound",setup+".bound.0")

optfile = open(setup+".opt","r")
optlines = optfile.readlines()
optfile.close()

optfile = open(setup+".opt","w")

write_newlines = True
for line in optlines:
    if re.match("FARGO_OPT\s*\+=",line) and write_newlines:
        optfile.write(newlines)
        write_newlines = False 
    optfile.write(line)
optfile.close()

cifile = open("condinit.c","r")
cilines = cifile.readlines()
cifile.close()

cifile = open("condinit.c","w")

for line in cilines:
    if re.match("void CondInit\s*\(",line):
        line = line.replace("CondInit","Init")
    cifile.write(line)

cifile.write(newcilines)
