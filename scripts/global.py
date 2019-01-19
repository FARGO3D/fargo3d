import re
import os

SRCDIR = "../src/"

try:
    global_file = open(SRCDIR+"global.h",'r')
    GLOBAL =  global_file.readlines()
    global_file.close()
except:
    print "(global.py) Error: global.h cannot be opened"
    exit()
globalex = open('global_ex.h', 'w')
warning = """ /* This file was created automatically during compilation
from global.h. Do not edit. See python script
"scripts/global.py" for details. */ """
warning += "\n\n"
globalex.write(warning)

for line in GLOBAL:
    search = re.match("([^/=]*)[=|;]", line)
    if search:
        globalex.write("extern "+search.group(1).strip()+";\n")
        continue
    globalex.write(line)
globalex.close()
