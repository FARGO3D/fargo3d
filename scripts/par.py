import re
import os
import argparse
import textwrap

parser = argparse.ArgumentParser(prog='par.py',
    description='Compile parameter definitions to var.c',
    epilog='This script is part of FARGO3D.')
parser.add_argument('params')
parser.add_argument('defaults')
parser.add_argument('-m', '--mandatories')
parser.add_argument('-o', '--output', default='var.c')
args = parser.parse_args()

def extract_variable_and_type(line):
    skip = re.match(r'\s*(?:#+)|($)', line)
    if skip is not None: return None

    varname = re.search(r'(\w+)\s', line)
    if varname is not None:
        realvar = re.match(r'(\w+)\s+(\+?-?\d+\.\d+e?[+-]?\d*)', line)
        if realvar is not None:
            name  = realvar.group(1).upper()
            value = realvar.group(2)
            return name, value, 'REAL'

        intvar = re.match(r'(\w+)\s+(\+?-?\d+)\s+', line)
        if intvar is not None:
            name  = intvar.group(1).upper()
            value = intvar.group(2)
            return name, value, 'INT'

        boolvar = re.match(r'(\w+)\s+(\w+)', line)
        if boolvar is not None:
            name  = boolvar.group(1).upper()
            value = boolvar.group(2).lower()
            if value in ['yes', 'no']:
                value = dict(no=0, yes=1).get(value)
                return name, value, 'BOOL'

        strvar = re.match(r'(\w+)\s+(.*)\s?', line)
        if strvar is not None:
            name  = strvar.group(1).upper()
            value = strvar.group(2)
            return name, value, 'STRING'

    raise NotImplementedError(f'Could not parse line: {line}')

with open(args.params, 'r') as params, open(args.defaults, 'r') as defaults, \
        open(args.output, 'w') as output:
    p = dict()

    # initialize with default parameters
    for line in defaults:
        tup = extract_variable_and_type(line)
        if tup is None:
            # skipping this line
            continue
        else:
            name, value, dtype = tup
            p[name] = value, dtype, False

    # override with specific parameters
    for line in params:
        tup = extract_variable_and_type(line)
        if tup is None:
            # skipping this line
            continue
        else:
            name, value, dtype = tup
            p[name] = value, dtype, False

    # update with mandatory parameters
    if os.path.isfile(args.mandatories):
        with open(args.mandatories, 'r') as mandatories:
            for line in mandatories:
                if re.match(r'\s*#+', line) is not None: continue
                name = re.search(r'(\w+)\s', line)
                if name not in p.keys():
                    raise ValueError('Mandatory parameter {} is not defined')
                value, dtype, _ = p[name]
                p[name] = value, dtype, True

    header = '''\
    #define __LOCAL
    #include "fargo3d.h"
    #undef __LOCAL

    void InitVariables() {
    '''
    output.write(textwrap.dedent(header))

    for name, (value, dtype, reqd) in p.items():
        need = 'YES' if reqd else 'NO'
        line = f'  init_var("{name}", (char*)&{name}, {dtype}, {need}, "{value}");\n'
        output.write(line)

    output.write('}')
