#!/usr/bin/env python

import re
import sys
import os

"""
Warning: Do not forget to include all the new flags in the .defaulflags file.
"""

BIN           = "bin"
try:
    os.chdir(BIN)                    #We are in the bin Folder. All references are from here.
except OSError:
    os.system("mkdir bin")
    os.chdir(BIN)

STDDIR     = "../std/"
SCRIPTSDIR = "../scripts/"
SRCDIR     = "../src/"

def get_defaults(filename=STDDIR+"defaultflags"):
    params = {}
    lines = open(filename,'r').readlines()
    for line in lines:
        try:
            split = line.split("=")
            params[split[0].strip()] = split[1].strip()
        except IndexError:
            continue
    return params

def get_last(filename=STDDIR+".lastflags"):
    params = {}
    try:
        lines = open(filename,'r').readlines()
    except IOError:
#        print "Using the default flags..."
        return 0
    for line in lines:
        try:
            split = line.split("=")
            params[split[0].strip()] = split[1].strip()
        except IndexError:
            continue
    return params

def get_variable(parameters):
    params = {}
    for parameter in parameters:
        try:
            split = parameter.split("=")
            params[split[0]] = split[1]
        except IndexError:
            print "Skipping", parameter
    return params

def complete_parameters(base,params):
    if parameters == None:
        return base
    else:
        for key in base.keys():
            if params.has_key(key):
                continue
            else:
                params[key] = base[key]
    return params

def check_coherence(base,params):
    for key in params.keys():
            if base.has_key(key):
                continue
            else:
                params.pop(key)
    return params

def build_is_fresh(base,params):
    for key in base.keys():
        if params[key] != base[key]:
            return 0
    return 1
    
def write_last(parameters):
    last = open(STDDIR+".lastflags",'w')
    for key in parameters.keys():
        last.write(key+"="+parameters[key]+"\n")
    last.close()

def clean_check(parameters, makeline):
    for param in parameters:
        if re.match("clean",param):
            os.system(makeline + "-s clean")
            exit()

def mrproper_check(parameters, makeline):
    for param in parameters:
        if re.match("mrproper",param):
            os.system(makeline + "-s mrproper")
            exit()

def jobs_check(parameters):
    for param in parameters:
        njobs = re.match("jobs=(\d+)\s*",param.lower())
        if njobs:
            return njobs.group(1)
    return None

make = "make -f "+ SRCDIR+ "makefile PYTHONMAKE=1 "
last = get_last()

#Checking for a clean...
clean_check(sys.argv, make)
mrproper_check(sys.argv, make)
njobs = jobs_check(sys.argv) #Number of threads
if njobs != None:
    njobs = int(njobs)
else:
    njobs = 8

base = last
if not last:
    base = get_defaults()
if len(sys.argv[1:]) > 0:
    parameters = get_variable(sys.argv[1:])
else:
    parameters = None

new_params = complete_parameters(base,parameters)
final_params = check_coherence(base,new_params)

write_last(final_params)

if not build_is_fresh(base,final_params):
    os.system(make + "clean")

line = make + "var.c "
for key in final_params.keys():
    line += key + "=" + final_params[key] + " "
os.system(line+">.tmp")

os.system("python "+ SCRIPTSDIR +"param.py")
os.system("python "+ SCRIPTSDIR +"global.py")

line = make + " rescale.c "
for key in final_params.keys():
    line += key + "=" + final_params[key] + " "
os.system(line)

line = make + " -j{0:d} ".format(njobs)
for key in final_params.keys():
    line += key + "=" + final_params[key] + " "

line += " allp"

os.system(line)
