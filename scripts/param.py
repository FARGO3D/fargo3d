import re  # Import Regex module
import time
import os

#Open & read var.c. Everything is stored in VAR.

try:
    var = open('var.c', 'r')
    VAR = var.readlines()
    var.close()
except:
    print "(param.py) Error: var.c cannot be opened"
    exit()

param = open('param.h', 'w')
warning = """ /*This file was created automatically during compilation
from var.c. Do not edit. See python script
"param.py" for details.*/ """
warning += "\n\n"

param.write(warning)
externals = ("#ifdef __LOCAL\n" + 
             "#define PARAM_EXTERNAL   \n" + 
             "#else\n" +
             "#define PARAM_EXTERNAL extern\n" + 
             "#endif\n\n")
param.write(externals)
for line in VAR:

    v_name = re.search("\(\"(\S+)\"",line)       # variable name
    v_type = re.search("\",\s\S+,\s(\S+),",line) # variable type
    v_referenced = re.search("\",\s(\S+)", line) #reference type for variable
    if(v_name):
        if(v_type.group(1) == "STRING"):
            new_line = "char " + v_name.group(1) + "[512]"
        if(v_type.group(1) == "INT"):
            new_line = "int  " + v_name.group(1)
        if(v_type.group(1) == "REAL"):
            new_line = "real " + v_name.group(1)
        if(v_type.group(1) == "BOOL"):
            new_line = "boolean " + v_name.group(1)
        new_line += ";\n"
        new_line = "PARAM_EXTERNAL " + new_line
        param.write(new_line)

param.close()
