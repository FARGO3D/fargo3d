import re
import os
import sys

def get_pardata(parname):

    parfile = open(parname,'r')
    par = parfile.readlines()
    parfile.close()

    realvariables = {}
    intvariables  = {}
    boolvariables = {}
    strvariables  = {}

    for line in par:
        skip = re.match("\s*#+",line)
        if skip != None:
            continue    
        varname = re.search("(\w+)\s",line)
        if varname != None:
            realvar = re.match("(\w+)\s+(\+?-?\d+\.\d+e?[+-]?\d*)",line)
            if realvar != None:
                name  = realvar.group(1).upper()
                value = realvar.group(2)
                realvariables[name] = value
                continue
            intvar = re.match("(\w+)\s+(\+?-?\d+)\s+",line)
            if intvar != None:
                name  = intvar.group(1).upper()
                value = intvar.group(2)
                intvariables[name] = value
                continue

            boolvar = re.match("(\w+)\s+(\w+)",line)
            if boolvar != None:
                name  = boolvar.group(1).upper()
                value = boolvar.group(2).lower()
                if value == "no" or value == "yes":
                    if value == "yes":
                        value = str(1)
                    elif value == "no":
                        value = str(0)
                    boolvariables[name] = value
                    continue
                    
            strvar = re.match("(\w+)\s+(.*)\s?",line)
            if strvar != None:
                name  = strvar.group(1).upper()
                value = strvar.group(2)
                strvariables[name] = value
                continue

    return  (realvariables,
             intvariables,
             boolvariables,
             strvariables)

def make_excess(active,default):
    """
    The result is stored in default. (pop method!)
    At the end, default has the non common parameters
    """
    for key in active:
        try:
            default.pop(key)
        except:
            continue

def make_varc(varc, parameters, partype, mand, init=False, end=False):
    """
    varc is a tuple of lines of var.c
    variables is a dict of parameters
    partype is "REAL", "INT", "STRING" or "BOOL".
    """
    if init:
        includes = '#define __LOCAL\n#include "../src/fargo3d.h"\n#undef __LOCAL\n\n'
        varc.append(includes)
        varc.append("void InitVariables() {\n")
    for parname in parameters:
        need = "NO"
        if mand != None:
            if mand.has_key(parname):
                need = "YES"
        parpoint = "(char*)" + "&" + parname
        new_line = "  init_var(" + '"' + parname + '"' + ", " + parpoint + \
            ", " + partype + ", " + need + ", " + '"' + parameters[parname] + \
            '");\n'
        varc.append(new_line)
    if end:
        varc.append("}")        

def get_mandatories(filename):
    mandfile = open(filename,'r')
    mandatory = mandfile.readlines()
    mandfile.close()
    mandatories = {}
    
    for line in mandatory:
        skip = re.match("\s*#+",line)
        if skip != None:
            continue
        varname = re.search("(\w+)\s",line)
        if varname != None:
            mandatories[varname.group(0).strip().upper()] = "M"
    return mandatories

if __name__ == "__main__":
    parname = sys.argv[1]
    def_parname = sys.argv[2]
    mandname = parname[:-3]+"mandatories"
    
    try:
        mandatories = get_mandatories(mandname)
    except IOError:
        mandatories = None
        print "You have not defined mandatory variables..."
        
    real, integer, boolean, string = get_pardata(parname) #SETUP.par
    def_real, def_integer, def_boolean, def_string = get_pardata(def_parname) #Default par
     
    make_excess(integer,mandatories)
    
    make_excess(real,def_real)
    make_excess(integer,def_integer)
    make_excess(boolean,def_boolean)
    make_excess(string,def_string)
    
    varc = []
    prolog = []
    epilog = []
    make_varc(prolog,[]       ,"FOO"   ,mandatories, init=True)
    make_varc(varc,real       ,"REAL"  ,mandatories)
    make_varc(varc,integer    ,"INT"   ,mandatories)
    make_varc(varc,string     ,"STRING",mandatories)
    make_varc(varc,boolean    ,"BOOL"  ,mandatories)
    make_varc(varc,def_real   ,"REAL"  ,mandatories)
    make_varc(varc,def_integer,"INT"   ,mandatories)
    make_varc(varc,def_string ,"STRING",mandatories)
    make_varc(varc,def_boolean,"BOOL"  ,mandatories)
    make_varc(epilog,[]       ,"FOO"   ,mandatories, end=True)

    var = open('var.c', 'w')
    for line in prolog+sorted(varc)+epilog:
        var.write(line)
    var.close()
        
