import re
import copy
import sys
import os

from par import get_pardata

STDDIR = "../std/" #Where is "boundary_template.c"
BINDIR = "../src/" #previously "../bin" but it makes more sense to write it to "../src"
SETUPDIR = "../setups/"

def pformat_read(name):
    try:
        with open(name, 'r', encoding='utf-8') as datafile:
            lines = datafile.readlines()
    except OSError:
        return {}
    field_name = ''
    field = {}
    for line in lines:
        search = re.match(r"\s*(\w+)\s*.*=\s*(.*)",line)  
        if search != None:
            field_name = search.group(1)
            field_rule = search.group(2)
            rule = field_name+" *= "+field_rule+";\n"
            field[field_name] = rule
    return field


SETUPNAME = sys.argv[1]
fieldstd = pformat_read ("../std/standard.units")
fieldsetup = pformat_read ("../setups/"+SETUPNAME+"/"+SETUPNAME+".units")
var = get_pardata ("../setups/"+SETUPNAME+"/"+SETUPNAME+".par")
realvar = var[0]
output = """
#include "fargo3d.h"

void rescale () {
"""
try:
    with open(BINDIR+"/rescale.c", 'w', encoding='utf-8') as rescale:
        rescale.write(output)
        for key in realvar:
            if key in fieldsetup:
                rescale.write(fieldsetup[key])
            else:
                if key in fieldstd:
                    rescale.write(fieldstd[key])
                else:
                    if ((key != 'XMIN') and (key != 'XMAX') \
                        and (key != 'YMIN') and (key != 'YMAX') \
                        and (key != 'ZMIN') and (key != 'ZMAX')):
                        print("Warning ! Scaling rule not found for "+key+". Is it dimensionless ?")
        rescale.write("}\n")
except OSError:
    print("Could not open rescale.c in "+BINDIR)
    exit()

