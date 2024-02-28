import re
import argparse
import textwrap

parser = argparse.ArgumentParser(prog='global.py',
    description='Compile global.h to global_ex.h',
    epilog='This script is part of FARGO3D.')
parser.add_argument('input')
parser.add_argument('-o', '--output', default='global_ex.h')
args = parser.parse_args()

with open(args.input, 'r') as gin, open(args.output, 'w') as gout:
    warning = f'''\
        /* This file was created automatically during compilation from
        {args.input}. Do not edit. See python script 'global.py' for details. */
        '''
    gout.write(textwrap.fill(textwrap.dedent(warning)) + '\n\n')

    for line in gin:
        m = re.match('([^/=]*)[=|;]', line)
        if m is not None:
            gout.write('extern ' + m.group(1).strip() + ';\n')
        else:
            gout.write(line)
