import re
import os

SRCDIR = "../src/"

try:
    with open(SRCDIR+"global.h", 'r', encoding='utf-8') as global_file:
        GLOBAL = global_file.readlines()
except OSError:
    print("(global.py) Error: global.h cannot be opened")
    exit()

warning = """ /* This file was created automatically during compilation
from global.h. Do not edit. See python script
"scripts/global.py" for details. */ """
warning += "\n\n"

with open('global_ex.h', 'w', encoding='utf-8') as globalex:
    globalex.write(warning)
    for line in GLOBAL:
        search = re.match("([^/=]*)[=|;]", line)
        if search:
            globalex.write("extern "+search.group(1).strip()+";\n")
            continue
        globalex.write(line)
