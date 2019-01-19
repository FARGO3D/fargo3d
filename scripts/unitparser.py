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
        datafile = open(name,'r')
    except IOError:
        return {}
    lines = datafile.readlines()
    datafile.close()
    field_name = ''
    field = {}
    for line in lines:
        search = re.match("\s*(\w+)\s*.*=\s*(.*)",line)  
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
try:
    rescale = open(BINDIR+"/rescale.c","w")
except IOError:
    print "Could not open rescale.c in "+BINDIR
    exit()
output = """
#include "fargo3d.h"

void rescale () {
"""
rescale.write(output)
for key in realvar:
    if fieldsetup.has_key(key):
        rescale.write(fieldsetup[key])
    else:
        if fieldstd.has_key(key):
            rescale.write(fieldstd[key])
        else:
            if ((key != 'XMIN') and (key != 'XMAX') \
                and (key != 'YMIN') and (key != 'YMAX') \
                and (key != 'ZMIN') and (key != 'ZMAX')):
                print "Warning ! Scaling rule not found for "+key+". Is it dimensionless ?"
rescale.write("}\n")
rescale.close()

