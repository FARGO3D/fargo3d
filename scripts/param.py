import re
import argparse
import textwrap

parser = argparse.ArgumentParser(prog='param.py',
    description='Compile var.c to params.h',
    epilog='This script is part of FARGO3D.')
parser.add_argument('input')
parser.add_argument('-o', '--output', default='params.h')
args = parser.parse_args()

with open(args.input, 'r') as var, open(args.output, 'w') as params:
    warning = '''\
    /* This file was created automatically during
    compilation from {}. Do not edit. See python script 'param.py'
    for details. */
    '''.format(args.input)
    params.write(textwrap.fill(textwrap.dedent(warning)) + '\n\n')

    externals = '''\
        #ifdef __LOCAL
        #define PARAM_EXTERNAL
        #else
        #define PARAM_EXTERNAL extern
        #endif
        '''
    params.write(externals)

    for line in var:
        v_name = re.search(r'\("(\S+)"', line)          # variable name
        v_type = re.search(r'",\s\S+,\s(\S+),', line)  # variable type
        v_referenced = re.search('",\s(\S+)',  line)    # reference type for variable

        if v_name is not None:
            if v_type.group(1) == 'STRING':
                new_line = 'char ' + v_name.group(1) + '[BUFSIZ]'
            if v_type.group(1) == 'INT':
                new_line = 'int  ' + v_name.group(1)
            if v_type.group(1) == 'REAL':
                new_line = 'real ' + v_name.group(1)
            if v_type.group(1) == 'BOOL':
                new_line = 'boolean ' + v_name.group(1)
            new_line += ';\n'
            new_line = 'PARAM_EXTERNAL ' + new_line
            params.write(new_line)
